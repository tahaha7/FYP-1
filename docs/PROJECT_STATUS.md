# EchoMind — Project Status (FYP Milestone Tracker)

**Project:** EchoMind — local, privacy-first meeting intelligence for Urdu / English code-switched meetings
**Student:** Taha Mehmood, BS Data Science, Final Year Project
**Repository:** tahaha7/FYP-1

This tracker is kept honest on purpose: it separates what is finished and
checked from what is written but still waiting on a GPU run. A supervisor
reading this should be able to see exactly where the project stands.

---

## 1. Phases

| Phase | Status | Deliverable |
|---|---|---|
| 1. Proposal & literature | Complete | Project goal and end-to-end pipeline flow (proposal text); dataset verified — ASLP-lab/UrduSpeech on Hugging Face, 156h, CC-BY-4.0, code-switched subset US-CS 89.4h; prior work on Urdu ASR and code-switching identified for the literature review |
| 2. Data pipeline | Code complete — verification in progress | `echomind/download_data.py` (corpus and benchmark downloaders, matched to the dataset's real folder layout) and `echomind/prepare_data.py` (train / val / test manifests, speaker-aware splits); a real download-and-prepare run on a corpus sample is the verification step |
| 3. Baseline evaluation | In progress | CPU smoke test DONE 23 Oct 2026: stock Whisper base on 13 real clips, WER 80.7% / CER 65.7% — pipeline proven end to end, detail in `docs/BASELINE_SMOKE_TEST.md`. The reportable baseline — stock Whisper Large-v3 on the US-benchmark manifest — runs on the university GPU PC under the protocol in `docs/EVALUATION.md` |
| 4. Fine-tuning | Code ready — awaiting GPU run | `scripts/train_lora.py` (Whisper Large-v3 + LoRA/PEFT, standard Hugging Face recipe), `scripts/merge_lora.py`, `scripts/convert_ct2.py`, and a step-by-step `FINETUNING_GUIDE.md` written for the university GPU PCs |
| 5. Full meeting pipeline | Code complete | `echomind/orchestrator.py` (transcribe → diarize → align → understand → evidence validation → minutes), plus `transcribe.py`, `diarize.py`, `speaker_id.py`, `understand.py`, `validate_evidence.py`, `name_resolve.py`, `procurement.py`, `storage.py` |
| 6. Evaluation & report | Pending | Baseline-vs-fine-tuned WER / CER results on the US-benchmark (results table in `docs/EVALUATION.md`, all cells TBD until the runs happen), error analysis, and the final FYP report write-up |

---

## 2. Verified vs awaiting GPU

### Checked and working (CPU machine, no model training involved)

- [x] Project structure, configuration (`config.yaml`) and split
  CPU / GPU requirements
- [x] Dataset download and manifest code, written against the dataset's
  verified real layout (`corpus/` for training, `benchmark/` for the
  hand-checked set)
- [x] All Python modules compile cleanly
- [x] Evaluation and data integrity fixes (23 Oct 2026): benchmark kept
  in its own folder and used whole as a test-only manifest, mixing with
  corpus data refused, reproducible speaker-aware splits by default,
  corpus-level WER/CER with a shared Urdu normaliser, `--language` option,
  run metadata saved with every result, pinned dependencies, training
  features computed on the fly. Tested in `tests/test_eval_and_data.py`;
  the train → merge → convert → evaluate chain was run end to end on a
  tiny random Whisper model to check the scripts (not a result)
- [x] Unit tests for the pure-Python logic pass — evidence validation,
  rule-based extraction, procurement scanning, name resolution and
  speaker-aware data splitting (`tests/test_modules.py`, run in CI on
  every push)
- [x] Encrypted local storage helpers (Fernet) implemented and exercised
  by the pipeline code path
- [x] Fine-tuning, merge and conversion scripts written following the
  standard Hugging Face Whisper + LoRA recipe

### Written, but awaiting a GPU run to be proven

- [ ] Stock Whisper Large-v3 baseline WER / CER on the US-benchmark
  (the "before" numbers)
- [ ] LoRA fine-tuning run on a university GPU PC, with checkpoints and
  per-epoch WER from the training script
- [ ] Merge + CTranslate2 conversion of the fine-tuned model, and the
  "after" evaluation on the identical manifest and settings
- [ ] pyannote diarization run (needs a GPU and an `HF_TOKEN` with the
  gated model's terms accepted); until then the pipeline runs with the
  single-speaker fallback
- [ ] ECAPA-TDNN enrolment and identification run on real voice samples
- [ ] End-to-end meeting run on a real recorded meeting, with minutes
  produced by the orchestrator

### Not yet built (planned, in order)

- [ ] DER evaluation harness (needs reference speaker segments)
- [x] EER / FAR / FRR trial harness for speaker identification
  (`python -m echomind.speaker_id eval`; still needs real trial clips)
- [ ] Precision / recall scoring for decisions and action items (needs
  human-labelled meetings)
- [ ] EchoMind Meeting Benchmark — a small set of real, consented meeting
  recordings for end-to-end testing
- [ ] Wiring speaker identification, name resolution and the procurement
  module fully into the orchestrator output
- [ ] Optional denoising (DeepFilterNet / RNNoise) if real recordings
  show it is needed; NeMo Sortformer comparison if pyannote diarization
  underperforms

---

## 3. How to read the results when they arrive

- Baseline and fine-tuned scores are always reported **together**, on the
  same benchmark manifest, from `docs/EVALUATION.md`.
- Any number produced on a CPU smoke test is a pipeline check, never a
  project result.
- Every status line in this file should be updated in the same commit as
  the work it describes — if a box is ticked, the evidence (output file,
  results JSON, or test run) is in the repository.
