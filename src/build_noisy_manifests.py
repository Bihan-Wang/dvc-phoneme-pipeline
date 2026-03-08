#!/usr/bin/env python3
"""Generate noisy audio variants and corresponding manifests from a clean manifest."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import tempfile
import wave
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--snr-levels", required=True, help="Python-list-like string, e.g. [40,30,20]")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--manifest-dir", required=True)
    return parser.parse_args()


def parse_snr_levels(value: str) -> list[float]:
    value = value.strip()
    if not value:
        raise ValueError("--snr-levels cannot be empty")

    if value.startswith("["):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("--snr-levels list must be non-empty")
        items = parsed
    else:
        items = [x.strip() for x in value.split(",") if x.strip()]
        if not items:
            raise ValueError("--snr-levels CSV must be non-empty")

    snrs: list[float] = []
    for x in items:
        if not isinstance(x, (int, float, str)):
            raise ValueError(f"Invalid SNR value: {x!r}")
        snrs.append(float(x))
    return snrs


def load_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty clean manifest: {path}")
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def utt_seed(base_seed: int, utt_id: str, snr_db: float) -> int:
    key = f"{base_seed}:{utt_id}:{snr_db:.6f}".encode("utf-8")
    return int.from_bytes(hashlib.md5(key).digest()[:8], byteorder="little", signed=False)


def snr_tag(snr_db: float) -> str:
    if snr_db.is_integer():
        iv = int(snr_db)
        sign = "p" if iv >= 0 else "m"
        return f"{sign}{abs(iv):02d}"
    value = str(snr_db).replace("-", "m").replace(".", "_")
    return f"x{value}"


def read_wav_mono_float(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        if n_channels != 1:
            raise ValueError(f"Only mono wav is supported: {path}")
        sampwidth = wf.getsampwidth()
        if sampwidth not in (1, 2, 4):
            raise ValueError(f"Unsupported sample width ({sampwidth}) for {path}")
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        pcm_bytes = wf.readframes(n_frames)

    if sampwidth == 1:
        x = np.frombuffer(pcm_bytes, dtype=np.uint8).astype(np.float32)
        signal = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32)
        signal = x / 32768.0
    else:
        x = np.frombuffer(pcm_bytes, dtype="<i4").astype(np.float32)
        signal = x / 2147483648.0

    return signal, sr, sampwidth


def write_wav_mono_float(path: Path, signal: np.ndarray, sr: int, sampwidth: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal, -1.0, 0.9999695)

    if sampwidth == 1:
        pcm = np.round(clipped * 128.0 + 128.0).astype(np.uint8)
    elif sampwidth == 2:
        pcm = np.round(clipped * 32768.0).astype("<i2")
    else:
        pcm = np.round(clipped * 2147483648.0).astype("<i4")

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with wave.open(str(tmp_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def add_noise(signal: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    signal_power = float(np.mean(signal * signal))
    if signal_power <= 0.0:
        return signal.copy()

    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise


def build_noisy_manifests(
    clean_manifest: Path,
    snr_levels: list[float],
    base_seed: int,
    audio_root: Path,
    manifest_dir: Path,
) -> None:
    rows = load_manifest(clean_manifest)

    # Ensure deterministic outputs without stale files from previous runs.
    if audio_root.exists():
        shutil.rmtree(audio_root)
    if manifest_dir.exists():
        shutil.rmtree(manifest_dir)
    audio_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for snr in snr_levels:
        tag = snr_tag(snr)
        snr_audio_dir = audio_root / f"snr_{tag}"
        snr_manifest_rows: list[dict] = []

        for row in rows:
            src_path = Path(row["wav_path"])
            if not src_path.is_absolute():
                src_path = (Path.cwd() / src_path).resolve()
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source wav in clean manifest: {src_path}")

            signal, sr, sampwidth = read_wav_mono_float(src_path)
            seed = utt_seed(base_seed, row["utt_id"], snr)
            noisy = add_noise(signal, snr, seed)

            dst_path = snr_audio_dir / f"{row['utt_id']}.wav"
            write_wav_mono_float(dst_path, noisy, sr, sampwidth)
            dst_rel = dst_path.resolve().relative_to(Path.cwd().resolve())

            snr_manifest_rows.append(
                {
                    **row,
                    "wav_path": str(dst_rel),
                    "audio_md5": file_md5(dst_path),
                    "sr": sr,
                    "duration_s": round(len(noisy) / float(sr), 6),
                    "snr_db": snr,
                }
            )

        snr_manifest_rows.sort(key=lambda x: x["utt_id"])
        manifest_path = manifest_dir / f"snr_{tag}.jsonl"
        atomic_write_jsonl(manifest_path, snr_manifest_rows)


def main() -> None:
    args = parse_args()
    build_noisy_manifests(
        clean_manifest=Path(args.clean_manifest),
        snr_levels=parse_snr_levels(args.snr_levels),
        base_seed=args.seed,
        audio_root=Path(args.audio_root),
        manifest_dir=Path(args.manifest_dir),
    )


if __name__ == "__main__":
    main()
