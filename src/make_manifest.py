import json
from pathlib import Path

out_path = Path("manifests/en/clean.jsonl")
wav_dir = Path("data/raw/en/wav")

samples = [
    {
        "stem": "demo_001",
        "text": "hello world"
    },
    {
        "stem": "demo_002",
        "text": "good morning"
    }
]

out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for s in samples:
        row = {
            "utt_id": f"en_{s['stem']}",
            "lang": "en",
            "wav_path": str(wav_dir / f"{s['stem']}.wav"),
            "ref_text": s["text"],
            "ref_phon": "",
            "audio_md5": "",
            "sr": 16000,
            "duration_s": None,
            "snr_db": None
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")