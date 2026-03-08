#!/usr/bin/env python3
"""Run phoneme ASR inference from noisy manifests and write prediction manifests."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest-dir", default=None)
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--output-manifest-dir", default=None)
    parser.add_argument("--output-manifest", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    parser.add_argument("--manifest-glob", default="*.jsonl")
    parser.add_argument("--max-utts-per-manifest", type=int, default=None)
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_name, path)
    finally:
        tmp_path = Path(tmp_name)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def resample_if_needed(signal: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return signal.astype(np.float32)

    if signal.size == 0:
        return signal.astype(np.float32)

    src_len = signal.shape[0]
    dst_len = int(round(src_len * (target_sr / src_sr)))
    if dst_len <= 1:
        return signal[:1].astype(np.float32)

    src_x = np.linspace(0.0, 1.0, num=src_len, endpoint=True)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=True)
    return np.interp(dst_x, src_x, signal).astype(np.float32)


def load_audio_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return audio, int(sr)


def run_manifest(
    rows: list[dict],
    processor,
    model,
    device: torch.device,
    batch_size: int,
    max_utts: int | None,
) -> list[dict]:
    if max_utts is not None:
        rows = rows[:max_utts]

    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    out_rows: list[dict] = []

    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i : i + batch_size]
        batch_audio: list[np.ndarray] = []

        for row in batch_rows:
            wav_path = Path(row["wav_path"])
            if not wav_path.is_absolute():
                wav_path = (Path.cwd() / wav_path).resolve()
            audio, sr = load_audio_mono(wav_path)
            audio = resample_if_needed(audio, sr, target_sr)
            batch_audio.append(audio)

        inputs = processor(
            batch_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_phons = processor.batch_decode(pred_ids)

        for row, pred in zip(batch_rows, pred_phons):
            out = dict(row)
            out["pred_phon"] = pred.strip()
            out_rows.append(out)

    return out_rows


def main() -> None:
    args = parse_args()

    has_input_dir = args.input_manifest_dir is not None
    has_input_file = args.input_manifest is not None
    if has_input_dir == has_input_file:
        raise ValueError("Provide exactly one of --input-manifest-dir or --input-manifest")

    has_output_dir = args.output_manifest_dir is not None
    has_output_file = args.output_manifest is not None
    if has_output_dir == has_output_file:
        raise ValueError("Provide exactly one of --output-manifest-dir or --output-manifest")

    if has_input_dir != has_output_dir:
        raise ValueError("Directory input requires directory output, file input requires file output")

    if has_input_dir:
        in_dir = Path(args.input_manifest_dir)
        manifest_paths = sorted(in_dir.glob(args.manifest_glob))
        if not manifest_paths:
            raise FileNotFoundError(f"No manifests found in {in_dir} matching {args.manifest_glob}")
        output_paths = [Path(args.output_manifest_dir) / p.name for p in manifest_paths]
    else:
        manifest_path = Path(args.input_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Input manifest does not exist: {manifest_path}")
        manifest_paths = [manifest_path]
        output_paths = [Path(args.output_manifest)]

    device = choose_device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForCTC.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    for manifest_path, out_path in zip(manifest_paths, output_paths):
        rows = load_jsonl(manifest_path)
        pred_rows = run_manifest(
            rows=rows,
            processor=processor,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_utts=args.max_utts_per_manifest,
        )
        atomic_write_jsonl(out_path, pred_rows)


if __name__ == "__main__":
    main()
