# Baseline smoke test — 23 Oct 2026

This is a **pipeline verification test**, not the official project
baseline. Its job was to prove the whole chain works on real data:
download -> manifests -> transcription -> WER/CER scoring. It passed.

## Setup

- Machine: CPU only, 2 cores, 8 GB RAM (no GPU on this machine)
- Model: Whisper **base**, stock (not fine-tuned), faster-whisper, int8
- Data: 134 short clips downloaded as a sample — corpus/US-CS/short/poetry
  plus benchmark/US-benchmark-CS/short/INTERVIEWS
- Eval set: val + test splits of that sample (speaker-aware split),
  13 clips, file `data/manifests/smoke_eval.jsonl`
- Command: `python -m echomind.evaluate_asr --manifest data/manifests/smoke_eval.jsonl --model-size base`

## Results (13 clips)

Note: these numbers were computed before the evaluation fixes of
23 Oct 2026 — they are per-file averages without Urdu Unicode
normalisation, and the 134-clip sample mixed corpus and benchmark data in
one split. They are kept as the record of the pipeline check only; re-run
with the current `evaluate_asr.py` for a corpus-level figure.

| Metric | Value |
| --- | --- |
| WER | 0.807 (80.7%) |
| CER | 0.657 (65.7%) |
| RTF | 0.73 (CPU) |

## What the errors look like (why fine-tuning is needed)

Reading the actual outputs, the stock base model fails on Urdu
code-switched speech in three specific ways:

1. **Wrong script** — Urdu speech comes out in Devanagari script
   (e.g. reference "میاں نواز شریف..." was transcribed starting with
   Devanagari characters), which counts as fully wrong even when the
   sounds are close.
2. **Translation instead of transcription** — one clip whose reference
   mixes Urdu and English ("آپ کی choice پہ ہے... jewelry سے کر لو")
   was output as full English ("You can wear a plane and do it"),
   losing the original words.
3. **Script mixing inside one line** — Urdu script, Roman and even a
   Tamil character in a single output line.

These are exactly the failure modes fine-tuning on UrduSpeech (whose
transcripts are consistently in Urdu script) is meant to fix, and they
are worth describing in the project report's baseline section.

## What the official baseline still needs (GPU PC)

- Same protocol, but model = Whisper **large-v3** stock, device cuda
- Eval set = the benchmark manifest only (hand-checked transcripts),
  kept separate from all training data
- See `FINETUNING_GUIDE.md` Step 7 and `docs/EVALUATION.md`

## Bug found by this test

The first run crashed with
`TypeError: open() got an unexpected keyword argument 'metadata_errors'`
— a version clash between faster-whisper 1.2.1 and newer PyAV.
Fixed by pinning `av==12.3.0` in both requirements files, and documented
in the guide's Problems section.
