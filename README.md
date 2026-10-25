# EchoMind

Local, privacy-first meeting intelligence for Urdu / English code-switched meetings.

GitHub repo: **tahaha7/FYP-1** — this is Taha's final year project (FYP).

**New here? Read [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md) first** — it
explains fine-tuning in easy words, step by step, for the university GPU PCs.

Meetings in Pakistan switch between Urdu and English mid-sentence. Normal
transcription tools handle that badly, and cloud tools mean sending sensitive
meeting audio to someone else's server. EchoMind does the whole job on your
own machine: transcribe who said what, identify the speakers, and produce a
summary, decisions and action items that are checked against the transcript.

## End-to-end flow

1. **Audio in** — live recording or an uploaded file, with consent.
2. **Preprocessing** — Voice Activity Detection (Silero VAD, via
   faster-whisper's `vad_filter`) cuts the silence. Optional denoising later.
3. **Diarization + ASR (parallel)**
   - pyannote.audio works out who spoke when.
   - Faster-Whisper (Whisper Large-v3 fine-tuned for Urdu/English
     code-switching, converted with CTranslate2) transcribes the speech.
     No language is forced, so code-switching is preserved.
4. **Alignment** — transcript segments are matched to diarization segments
   by time overlap, giving a speaker-attributed transcript.
5. **Speaker identification** — ECAPA-TDNN voiceprints match diarized
   speakers against enrolled voices. Unmatched speakers stay as
   Speaker 1, Speaker 2, etc. An optional Name Resolution helper
   (`echomind/name_resolve.py`) can suggest names from context —
   marked as suggested, never confirmed.
6. **Meeting understanding** — an LLM (or the built-in rule-based fallback)
   extracts a summary, decisions, action items and sentiment.
7. **Evidence validation** — every extracted item must carry a quote and a
   timestamp from the actual transcript. Items whose evidence is not found
   in the transcript are rejected, not delivered.
8. **Storage + output** — minutes (`.md` + `.json`) are produced, and audio,
   transcripts, voiceprints and records are stored locally, encrypted with
   Fernet. No cloud by default.
9. **Optional Procurement Module** (`echomind/procurement.py`, basic
   rule-based version) — finds prices/quotes in the transcript and flags
   ones far from the median quote as unusual.

A plain-Python Graph Orchestrator (`echomind/orchestrator.py`) runs the
steps, retries failed steps twice, and records the status of each step.
NVIDIA NeMo Sortformer is noted as a backup diarization option for
difficult audio where pyannote struggles — it is not in Phase 1 code.

## Dataset

**ASLP-lab/UrduSpeech** — https://huggingface.co/datasets/ASLP-lab/UrduSpeech

- 156 hours, 71,792 utterances, CC-BY-4.0
- Real repo layout (checked against the actual files — the dataset card
  shows a different, tidier layout): training data is under `corpus/`,
  the benchmark under `benchmark/`.
- `corpus/US-CS` — Urdu-English code-switched, 89.4h — **main training source**
- `corpus/US-Std` — standard Pakistani Urdu, 59.2h
- `corpus/US-EngPk` — Pakistani-accented English, 7.3h
- Each subset has `short/` (<=10s) and `long/` folders with 12 categories
  (comedy_show, drama, film, food, interview, news, podcast, poetry,
  proses, roadside_interview, vlogs, youtube_review). Each category has
  `audio/*.wav` plus a `*_final_transcription.jsonl`.
- `benchmark/US-benchmark-CS` (and -Std / -EngPk) is the 9-hour manually
  checked set. Its category folders are uppercase (INTERVIEWS, NEWS, ...)
  with `.WAV` files and a `clean_transcription.jsonl` that also has a
  `ground_truth` field. `download_data.py --benchmark` and
  `prepare_data.py` both handle this layout.

Note on quality: the main corpus transcriptions are model-generated with
manual validation (reported 97.6% confidence). The benchmark set is the
better gold set for final WER/CER reporting — use the main corpus for
training, the benchmark for final evaluation.

Citation (from the dataset page):

```bibtex
@misc{haq2026urduspeech156hoururduspeech,
  title={UrduSpeech: A 156-Hour Urdu Speech Corpus with 12-Dimension Paralinguistic Annotations},
  author={Attia Nafees ul Haq and Zeyu Zhu and Jingbin Hu and ChunJiang He and Lei Xie},
  year={2026},
  eprint={2605.17846},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  url={https://arxiv.org/abs/2605.17846},
}
```

## Folder layout

```
echomind/
  README.md
  config.yaml
  requirements-cpu.txt      # light deps for the CPU smoke test
  requirements-gpu.txt      # full deps for training / diarization / speaker ID
  echomind/
    __init__.py
    download_data.py        # download a small sample (or more) from HF
    prepare_data.py         # build train/val/test manifests
    transcribe.py           # faster-whisper wrapper
    diarize.py              # pyannote diarization (GPU machine)
    speaker_id.py           # ECAPA-TDNN enrol / identify / EER (CPU or GPU)
    storage.py              # Fernet encrypted local storage
    orchestrator.py         # MeetingPipeline — runs all steps
    understand.py           # summary / decisions / action items
    validate_evidence.py    # checks claims against the transcript
    name_resolve.py         # suggests names for anonymous speakers (optional)
    procurement.py          # finds prices/quotes, flags unusual ones (optional)
    evaluate_asr.py         # WER / CER on a manifest
  scripts/
    train_lora.py           # LoRA fine-tune of whisper-large-v3 (GPU only)
    merge_lora.py           # merge the LoRA adapter into a full model
    convert_ct2.py          # convert merged model to Faster-Whisper format
  data/        (gitignored)  downloaded dataset + manifests
  models/      (gitignored)  converted / fine-tuned models
  outputs/     (gitignored)  minutes, evaluation results, encrypted records
```

## Setup

### Quick start on a university GPU PC

Full explanation in [FINETUNING_GUIDE.md](FINETUNING_GUIDE.md). Short version:

```bash
git clone https://github.com/tahaha7/FYP-1.git && cd FYP-1
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
python -m echomind.download_data --subset CS --format short --category all --max-categories 12
python -m echomind.prepare_data
python scripts/train_lora.py --batch-size 4 --grad-accum 8 --epochs 3
```

### CPU smoke test (this machine — no GPU)

```bash
pip install -r requirements-cpu.txt

# 1. small sample only: CS subset, short clips, interview category
python -m echomind.download_data

# 2. build manifests
python -m echomind.prepare_data

# 3. check ASR on the test split with the small 'base' model
python -m echomind.evaluate_asr --manifest data/manifests/test.jsonl --limit 10
```

### GPU machine (training, diarization, speaker ID)

A university GPU PC (16 GB+ VRAM, bf16-capable GPU recommended):

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
export HF_TOKEN=...   # needed for the gated pyannote model

# download more data, e.g. all short CS categories
python -m echomind.download_data --subset CS --format short --category all --max-categories 12

python -m echomind.prepare_data          # speaker-aware split by default

# the hand-checked benchmark: own folder, test-only manifest
python -m echomind.download_data --benchmark --subset CS --format short --category all --max-categories 12
python -m echomind.prepare_data --benchmark

# fine-tune - full guide: FINETUNING_GUIDE.md
python scripts/train_lora.py --batch-size 4 --grad-accum 8

# merge the adapter into a full model, then convert it for faster-whisper
python scripts/merge_lora.py
python scripts/convert_ct2.py --model models/whisper-merged --out models/whisper-lora-ct2
```

### Speaker enrolment (CPU or GPU)

ECAPA-TDNN is small, so this runs on a laptop too (it uses the GPU if there
is one). Get written consent before enrolling anyone. Record 3-5 clips per
person, at least 10 s each, on the same kind of mic used in meetings, and a
separate set of clips per person for testing (never used for enrolment).

```bash
pip install torch==2.5.1 torchaudio==2.5.1 speechbrain==1.0.2 cryptography numpy pyyaml

python -m echomind.speaker_id enroll --name rayyan data/enroll/rayyan/*.wav
python -m echomind.speaker_id list
python -m echomind.speaker_id identify data/verify/rayyan/clip1.wav
python -m echomind.speaker_id delete --name rayyan     # consent withdrawn

# FAR / FRR / EER: data/verify/<name>/*.wav, include someone NOT enrolled
python -m echomind.speaker_id eval data/verify
```

Voiceprints are averaged over the clips and saved encrypted in
`outputs/voiceprints/` with the key from `storage.key_path`. Enrolment
refuses clips under 10 s and warns about a clip that does not match the
person's other clips. `eval` prints the EER and the threshold at the EER
point; put it in `speaker_id.threshold` in `config.yaml`.

## Hardware — stated honestly

- The machine this scaffold was built on has **no GPU (2 CPU, 8GB RAM)**.
  Only the CPU smoke test above is feasible there, and only with a small
  faster-whisper model (`base` by default, configurable in `config.yaml`).
- **Fine-tuning Whisper Large-v3 with LoRA and running pyannote diarization
  are GPU jobs.** They run on the university's high-end GPU PCs. A 6 GB
  laptop GPU is not enough to fine-tune Large-v3. Nothing about GPU
  performance is claimed from the CPU machine.

## Metrics used

- ASR: WER / CER (`echomind/evaluate_asr.py`) — corpus-level (total errors
  over total reference words), after the shared Urdu/English normaliser in
  `echomind/text_norm.py`, broken down by subset and category. Every result
  file records the git commit, package versions, settings and manifest hash.
- Diarization: DER (roadmap — needs reference diarization labels)
- Speaker ID: EER / FAR / FRR (roadmap — needs an enrolment trial set)
- Extraction: precision / recall for decisions and action items (roadmap)
- Speed: real-time factor, latency, GPU/memory usage (roadmap)

## Roadmap

**Phase 1 (done — this scaffold)**
- Project structure, config, requirements split (CPU / GPU)
- Dataset download helper (small sample by default) + manifest builder
- faster-whisper transcription, pyannote diarization, ECAPA speaker ID code
- Encrypted local storage helpers
- Pipeline orchestrator with retries, rule-based understanding fallback,
  LLM prompt builder, evidence validation
- WER/CER evaluation script, LoRA training script, CTranslate2 convert script

**Next phases**
- Run the CPU smoke test and record baseline WER/CER on UrduSpeech-CS
- Enrolment UI / CLI flow for voiceprints; per-speaker identification in
  the pipeline (Phase 1 keeps diarization labels by default)
- DER and EER/FAR/FRR evaluation harnesses
- LoRA fine-tune run on GPU, convert, and re-evaluate vs the baseline
- Improve the basic Name Resolution and Procurement modules (currently
  simple rule-based versions in `name_resolve.py` / `procurement.py`),
  and wire them into the orchestrator
- A small consented-meetings benchmark (EchoMind Meeting Benchmark) for
  realistic end-to-end testing

## Licence notes

- Code in this repo: Taha's project, no licence chosen yet.
- UrduSpeech dataset: **CC-BY-4.0** — free to use, citation required
  (see above). Respect the dataset's own out-of-scope uses: do not use it
  to identify or track individuals without consent.
