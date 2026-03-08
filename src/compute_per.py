#!/usr/bin/env python3
"""Compute PER metrics for clean and noisy prediction manifests."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-pred-manifest", required=True)
    parser.add_argument("--noisy-pred-dir", required=True)
    parser.add_argument("--snr-levels", required=True, help="CSV list, e.g. 40,30,20,10,0")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def snr_tag(snr: float) -> str:
    if float(snr).is_integer():
        iv = int(snr)
        sign = "p" if iv >= 0 else "m"
        return f"{sign}{abs(iv):02d}"
    value = str(snr).replace("-", "m").replace(".", "_")
    return f"x{value}"


def parse_snr_csv(value: str) -> list[float]:
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise ValueError("Empty --snr-levels")
    return [float(x) for x in items]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def phon_units(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    s = re.sub(r"\s+", "", s)
    return list(s)


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev
            else:
                dp[j] = min(prev + 1, dp[j] + 1, dp[j - 1] + 1)
            prev = tmp
    return dp[m]


def per_from_rows(rows: list[dict]) -> dict:
    total_edits = 0
    total_ref = 0
    used = 0

    for row in rows:
        ref = phon_units(row.get("ref_phon", ""))
        hyp = phon_units(row.get("pred_phon", ""))
        if not ref:
            continue
        total_edits += edit_distance(ref, hyp)
        total_ref += len(ref)
        used += 1

    per = (total_edits / total_ref) if total_ref > 0 else None
    return {
        "utterances": used,
        "total_ref_units": total_ref,
        "total_edits": total_edits,
        "per": per,
    }


def main() -> None:
    args = parse_args()

    clean_rows = load_jsonl(Path(args.clean_pred_manifest))
    clean_stats = per_from_rows(clean_rows)

    by_snr: dict[str, dict] = {}
    snr_values = parse_snr_csv(args.snr_levels)
    per_values: list[float] = []

    for snr in snr_values:
        tag = snr_tag(snr)
        manifest_path = Path(args.noisy_pred_dir) / f"snr_{tag}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing noisy prediction manifest: {manifest_path}")
        stats = per_from_rows(load_jsonl(manifest_path))
        key = str(int(snr)) if float(snr).is_integer() else str(snr)
        by_snr[key] = stats
        if stats["per"] is not None:
            per_values.append(float(stats["per"]))

    payload = {
        "clean": clean_stats,
        "by_snr": by_snr,
        "mean_noisy_per": (sum(per_values) / len(per_values)) if per_values else None,
    }
    atomic_write_json(Path(args.output), payload)


if __name__ == "__main__":
    main()
