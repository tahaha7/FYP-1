# Fine-tuning guide — in easy words

## What fine-tuning means here

Whisper already knows how to turn speech into text, but it is not very good
at Urdu, and it really struggles when people mix Urdu and English in one
sentence. Fine-tuning means: we show Whisper many examples of Urdu and
code-switched speech *with the correct text*, and let it practice on them.
We use LoRA, which only trains a small part of the model instead of the
whole thing — so it fits on one GPU and finishes in hours, not days.

You do everything below on the university GPU PC. Run the commands from
inside the project folder, in order.

## What you need

- A PC with an NVIDIA GPU. Check it first (step 4 below).
- **VRAM (GPU memory): about 16 GB or more is recommended** for
  Whisper Large-v3 LoRA with batch size 4. On a 24 GB card you are
  comfortable. On a smaller card, use batch size 2 (step 8 explains how).
- Around 50 GB free disk space (dataset + model + checkpoints).
- Internet for the first download of the dataset and the base model.

## Step 1 — get the code

```bash
git clone https://github.com/tahaha7/FYP-1.git
cd FYP-1
```

## Step 2 — make a fresh python environment

This keeps this project's packages separate from everything else on the PC.

```bash
python -m venv .venv
```

Activate it:

- Windows: `.venv\Scripts\activate`
- Linux: `source .venv/bin/activate`

## Step 3 — install everything

First PyTorch with GPU support, then the rest of the project packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.txt
```

This download is big (a few GB). It is a one-time thing.

## Step 4 — check the GPU is visible

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

You want to see the GPU name in the first command, and `True` plus the GPU
name from the second. If it prints `False`, stop here — the PyTorch install
does not match the GPU/driver (see Problems below).

## Step 5 — download the dataset

Start really small to test everything works. The poetry category has
under 100 short clips, so it downloads in a couple of minutes:

```bash
python -m echomind.download_data --category poetry
```

Run Steps 6 and 7 with that first — if the pipeline works on poetry, it
works, and you have not spent hours downloading yet.

Then get the interview category (the most meeting-like data, about
5,500 short clips — this one is a few GB, it takes longer):

```bash
python -m echomind.download_data
```

When that works, download the full code-switched short set (all 12
categories). This is the main training data.

```bash
python -m echomind.download_data --subset CS --format short --category all --max-categories 12
```

Optional, later: also get the `long` clips the same way
(`--format long`), and the Std subset (`--subset Std`). Download one
subset at a time — the whole dataset is large.

## Step 6 — prepare the training files

This reads all the transcription files, matches each one with its audio,
and splits everything into train / val / test lists.

```bash
python -m echomind.prepare_data
```

It prints how many clips and hours are in each split. The lists are saved
in `data/manifests/` (`train.jsonl`, `val.jsonl`, `test.jsonl`).

## Step 7 — (recommended) get the "before" score

Before fine-tuning, check how good the *original* Whisper is on your test
set. This is the number your fine-tuned model has to beat, and it goes in
your report.

```bash
python -m echomind.evaluate_asr --manifest data/manifests/test.jsonl --model-size large-v3 --device cuda --compute-type float16 --limit 50
```

It prints WER and CER (lower = better) and saves them to
`outputs/eval_results.json`. Save a copy of this file as your baseline.

Short on time or VRAM? Use `--model-size base` for a quicker, rougher
baseline — just compare like with like later.

**For your final report numbers, use the benchmark set, not the test
split above.** The benchmark is the small part of UrduSpeech that humans
checked by hand, so its transcripts can be trusted. Keep it in a
*separate* folder so it never gets mixed into your training data:

```bash
# download the benchmark code-switched short set, into its own folder
python -m echomind.download_data --benchmark --subset CS --format short --category interview --out data/benchmark_raw

# make a manifest from just that folder
python -m echomind.prepare_data --data-dir data/benchmark_raw --out-dir data/benchmark_manifests

# score any model against it (use the test list it creates)
python -m echomind.evaluate_asr --manifest data/benchmark_manifests/test.jsonl --model-size large-v3 --device cuda --compute-type float16
```

Never train on the benchmark data — if the model has already seen the
test clips, the score means nothing.

## Step 8 — fine-tune

```bash
python scripts/train_lora.py --batch-size 4 --grad-accum 8 --epochs 3
```

What the settings mean, quickly:

- `--epochs 3` — the model sees the full training data 3 times.
- `--batch-size 4` — 4 clips are processed together. Bigger = faster but
  uses more GPU memory.
- `--grad-accum 8` — updates are averaged over 8 batches, which makes
  training steadier.

What you will see: the model and data load, then a progress bar with a
`loss` number that should slowly go **down**, and a `wer` score after each
epoch that should also go down. Checkpoints (saved copies of the model)
are written to `models/whisper-lora/` after every epoch.

How long: it depends on the GPU and how much data you downloaded. Do the
small test download first and one epoch to see the speed on your PC —
the full CS short set on a high-end GPU is a few hours, not minutes.

The result in `models/whisper-lora/` is a **LoRA adapter** — a small file
with only the changes, not the full model. The next two steps turn it
into a model you can actually use.

## Step 9 — merge the adapter into the full model

```bash
python scripts/merge_lora.py
```

This combines the base Whisper model + your adapter into one full model
in `models/whisper-merged/`.

## Step 10 — convert it for fast use

```bash
python scripts/convert_ct2.py --model models/whisper-merged --out models/whisper-lora-ct2
```

This converts the model to the CTranslate2 format that Faster-Whisper
runs. This is the model the meeting pipeline will use.

## Step 11 — get the "after" score

Run the same test as step 7, but with your converted model:

```bash
python -m echomind.evaluate_asr --manifest data/manifests/test.jsonl --model-size models/whisper-lora-ct2 --device cuda --compute-type float16 --limit 50
```

Compare the WER with your baseline from step 7. Fine-tuning worked if
this number is clearly lower.

## Step 12 — use your model in the pipeline

```bash
python -m echomind.orchestrator meeting.wav --model-size models/whisper-lora-ct2 --device cuda --compute-type float16
```

Minutes are written to `outputs/meetings/` as `.md` and `.json`.

## Problems you will hit

**CUDA out of memory** (the most common one)
The GPU ran out of memory. Lower the batch size and raise grad accum so
the effective training size stays similar:

```bash
python scripts/train_lora.py --batch-size 2 --grad-accum 16 --epochs 3
```

Also close other programs using the GPU. Still failing on a small card?
Train on the Std subset or fewer categories first.

**`torch.cuda.is_available()` prints False**
PyTorch cannot see the GPU. Usually the wrong PyTorch wheel or an old
GPU driver. Re-run step 3 exactly as written, and check `nvidia-smi`
works. On a university PC, you may need the lab admin for driver updates.

**`TypeError: open() got an unexpected keyword argument 'metadata_errors'`**
This is a version clash between faster-whisper and PyAV (the audio
library it uses) — newer PyAV removed something faster-whisper needs.
The requirements files already pin the working versions, so this means
something upgraded PyAV afterwards. Fix it with:

```bash
pip install av==12.3.0
```

**pyannote / HF_TOKEN errors**
You only need this for *diarization* (who spoke when) in the pipeline —
**not for training**. To use it: make a free Hugging Face account, open
the `pyannote/speaker-diarization-3.1` model page and accept its terms,
create an access token on the HF site, then set it before running the
pipeline: `export HF_TOKEN=your_token` (Windows: `set HF_TOKEN=your_token`).
Without it the pipeline still works — everyone is just labelled
SPEAKER_00.

**Training crashed halfway**
Checkpoints are saved after every epoch in `models/whisper-lora/`
(folders named `checkpoint-...`). To continue from the last one, open
`scripts/train_lora.py`, find the line `trainer.train()` near the end,
change it to `trainer.train(resume_from_checkpoint=True)`, and run the
training command again.

**"no train rows found"**
You skipped step 6, or you are not in the project folder. Run
`python -m echomind.prepare_data` from the `FYP-1` folder first.

**Downloads are slow or disk fills up**
The dataset is big. Download one subset/format at a time (step 5),
delete nothing you still need, and keep data, models and outputs on the
drive with the most free space.
