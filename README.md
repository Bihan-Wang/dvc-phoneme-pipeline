# DVC pipeline for phoneme ASR robustness to noise

This project follows the lab specification with manifests as the core contract.

## Progress (first 4 parts)

1. **Experiment config (`params.yaml`)**
   - Languages, SNR levels, seed
   - Dataset path/patterns
   - espeak-ng voice mapping
   - Manifest output path

2. **Git + DVC repo setup**
   - `.git/` and `.dvc/` are initialized
   - `dvc.yaml` now defines the first stage
   - Large audio/intermediate artifacts are ignored by Git

3. **Manifest contract**
   - JSONL line format
   - Required fields:
     - `utt_id` (stable across clean/noisy)
     - `lang`
     - `wav_path` (relative)
     - `ref_text`
     - `ref_phon` (from `espeak-ng`)
     - `audio_md5`
     - plus `sr`, `duration_s`, `snr_db`
   - Atomic creation: write temp file then `os.replace`

4. **Clean manifest + phoneme references**
   - Implemented in `src/build_clean_manifest.py`
   - Reads LibriSpeech `*.trans.txt`
   - Validates mono wav and extracts `sr` / `duration_s`
   - Computes `audio_md5`
   - Calls `espeak-ng` to generate `ref_phon`
   - Writes `manifests/en/clean.jsonl` atomically


5. **Noisy variants + noisy manifests**
   - Implemented in `src/build_noisy_manifests.py`
   - Reads `manifests/en/clean.jsonl` only (no folder scan)
   - Generates noisy wav for each `snr_levels` value
   - Produces one manifest per SNR under `manifests/en/noisy/`
   - Keeps `utt_id` invariant and updates `wav_path`, `audio_md5`, `snr_db`
   - Uses deterministic seed per (`seed`, `utt_id`, `snr_db`)

## Stage command

```bash
python3 src/build_clean_manifest.py \
  --lang en \
  --dataset-root LibriSpeech/dev-clean \
  --transcript-glob "*.trans.txt" \
  --audio-ext ".wav" \
  --espeak-voice en-us \
  --output manifests/en/clean.jsonl
```

Equivalent DVC stage: `make_clean_manifest_en` in `dvc.yaml`.
