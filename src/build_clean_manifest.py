#!/usr/bin/env python3
"""Build a clean manifest from LibriSpeech-style transcripts and wav files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
import wave


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", required=True, help="Language code, e.g. en")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory")
    parser.add_argument(
        "--transcript-glob",
        default="*.trans.txt",
        help="Glob for transcript files relative to dataset root",
    )
    parser.add_argument("--audio-ext", default=".wav", help="Audio extension to use")
    parser.add_argument("--espeak-voice", required=True, help="espeak-ng voice")
    parser.add_argument("--output", required=True, help="Output clean manifest path")
    return parser.parse_args()


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_wav_metadata(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        if channels != 1:
            raise ValueError(f"Expected mono wav, got {channels} channels: {path}")
        sr = wf.getframerate()
        nframes = wf.getnframes()
    duration_s = nframes / float(sr)
    return sr, duration_s


def to_phonemes(text: str, voice: str) -> str:
    cmd = ["espeak-ng", "-q", "--ipa=3", "-v", voice, text]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "espeak-ng is not installed or not in PATH; install it before running this stage."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"espeak-ng failed for text={text!r}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def iter_transcript_entries(transcript_path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Malformed transcript line in {transcript_path}: {line}")
            utt_stem, ref_text = parts
            entries.append((utt_stem, ref_text))
    return entries


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


def build_manifest(
    lang: str,
    dataset_root: Path,
    transcript_glob: str,
    audio_ext: str,
    espeak_voice: str,
    output_path: Path,
) -> None:
    transcript_files = sorted(dataset_root.rglob(transcript_glob))
    if not transcript_files:
        raise FileNotFoundError(
            f"No transcript files found under {dataset_root} with pattern {transcript_glob}"
        )

    rows: list[dict] = []
    for transcript_path in transcript_files:
        for utt_stem, ref_text in iter_transcript_entries(transcript_path):
            wav_path_abs = (transcript_path.parent / f"{utt_stem}{audio_ext}").resolve()
            if not wav_path_abs.exists():
                raise FileNotFoundError(f"Missing audio file for {utt_stem}: {wav_path_abs}")

            wav_rel = wav_path_abs.relative_to(Path.cwd().resolve())
            sr, duration_s = read_wav_metadata(wav_path_abs)
            row = {
                "utt_id": f"{lang}_{utt_stem}",
                "lang": lang,
                "wav_path": str(wav_rel),
                "ref_text": ref_text,
                "ref_phon": to_phonemes(ref_text, espeak_voice),
                "audio_md5": file_md5(wav_path_abs),
                "sr": sr,
                "duration_s": round(duration_s, 6),
                "snr_db": None,
            }
            rows.append(row)

    rows.sort(key=lambda x: x["utt_id"])
    atomic_write_jsonl(output_path, rows)


def main() -> None:
    args = parse_args()
    build_manifest(
        lang=args.lang,
        dataset_root=Path(args.dataset_root),
        transcript_glob=args.transcript_glob,
        audio_ext=args.audio_ext,
        espeak_voice=args.espeak_voice,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
