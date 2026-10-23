# EchoMind — Evaluation Methodology

This document defines how EchoMind is evaluated, before any results are
reported. The rule it follows throughout: **a score is only worth
reporting if someone else could reproduce it** — same data, same splits,
same settings, stated in advance.

---

## 1. Metrics

### 1.1 Word Error Rate (WER)

WER measures how different the system's transcript is from the correct
transcript, counted in words: wrong words (substitutions), missing words
(deletions) and extra words (insertions), divided by the number of words
in the correct transcript. It is computed with `jiwer` in
`echomind/evaluate_asr.py`, after normalising both sides the same way
(lowercasing, removing Urdu and English punctuation, collapsing spaces).

**Direction:** lower is better. 0.00 is a perfect transcript. As a rough
guide for read or broadcast speech, below 0.10 is strong, 0.10–0.25 is
usable, and above that needs work — spontaneous code-switched meeting
speech is harder, so EchoMind's scores are always reported next to the
stock model's score on the *same* data (Section 3), never on their own.

### 1.2 Character Error Rate (CER)

CER is the same idea as WER but counted in characters instead of words.
It matters for Urdu because word boundaries and spelling variation in
Urdu script can make WER look worse than the transcript really is — a
word spelled two acceptable ways counts as a full word error but only a
small character error. `evaluate_asr.py` reports CER alongside WER for
every run.

**Direction:** lower is better, and CER is normally well below WER. A
large gap between the two is itself a finding: it usually means spelling
or script variation, not missed content.

### 1.3 Diarization Error Rate (DER)

DER measures the "who spoke when" part of the system. It is the total
time the diarization is wrong — speech it missed, silence or noise it
labelled as speech, and speech it attributed to the wrong speaker —
divided by the total speech time. It is computed against reference
speaker segments, which UrduSpeech does not provide for meetings, so DER
is evaluated on the EchoMind Meeting Benchmark once that is collected.

**Direction:** lower is better. Published diarization systems on
realistic meeting audio commonly sit around 0.10–0.30; overlapping speech
is the hardest part and the first place to look when DER is high.

### 1.4 Equal Error Rate (EER), False Acceptance Rate (FAR), False Rejection Rate (FRR)

These three measure speaker identification (ECAPA-TDNN voiceprints). Each
identification attempt ends in a similarity score, and a threshold
decides "this is the enrolled speaker" or not. **FAR** is how often an
impostor is wrongly accepted; **FRR** is how often the genuine speaker is
wrongly rejected. Raising the threshold lowers FAR but raises FRR. The
**EER** is the error rate at the threshold where the two are equal — a
single number for comparing speaker-ID setups. Evaluation needs a trial
set: genuine attempts (a speaker's own clips) and impostor attempts
(other speakers' clips), run against the enrolled voiceprints.

**Direction:** lower is better for all three. Which side matters more
depends on use: for naming speakers in minutes, false acceptance (naming
the wrong person) is the more damaging error, so the operating threshold
in `config.yaml` (cosine 0.35) should be reviewed against measured FAR,
not kept by default. *Status: trial protocol defined here; the harness is
not yet built.*

### 1.5 Precision and Recall (decisions and action items)

These measure the Meeting Understanding output against human-labelled
minutes for the same meetings. **Precision** = of the decisions / action
items the system produced, how many are real. **Recall** = of the real
decisions / action items in the meeting, how many the system found. The
evidence validator affects precision directly: items it rejects because
their quote is not in the transcript are counted and reported, not
silently discarded, so the extraction quality can be analysed honestly.

**Direction:** higher is better for both, and they are reported together
(never precision alone — a system that outputs one safe item scores
perfect precision and near-zero recall). *Status: protocol defined here;
needs human-labelled meetings, so it is evaluated on the EchoMind Meeting
Benchmark.*

### 1.6 Real-Time Factor (RTF)

RTF is processing time divided by audio duration. An RTF of 0.25 means a
10-minute meeting was transcribed in 2.5 minutes. `evaluate_asr.py`
records RTF per file (wall-clock transcription time over clip duration)
and reports the average.

**Direction:** lower is better; below 1.0 means faster than real time.
RTF is meaningless without the hardware, so every reported RTF names the
machine and settings (GPU model, compute type — e.g. float16 on GPU vs
int8 on CPU).

---

## 2. Dataset protocol

1. **Train on the corpus, report on the benchmark.** Fine-tuning uses
   the UrduSpeech corpus (primarily US-CS). Final, reportable WER / CER
   numbers come from the manually checked **US-benchmark**, whose
   transcripts were corrected by hand and can be trusted as ground truth.
2. **Never train on the benchmark.** Benchmark data is downloaded to a
   separate folder (`data/benchmark_raw`) with its own manifests and is
   never merged into the training manifests. A model scored on data it
   trained on is measuring memory, not ability.
3. **Speaker-aware splits.** Corpus manifests are built with
   `prepare_data.py --speaker-aware`, so no speaker appears in more than
   one of train / val / test. Scores are therefore measured on voices the
   model has never heard.
4. **Fixed seed.** Splits use seed 42 (the default in `prepare_data.py`).
   Any change of seed or split is a change of experiment and is recorded
   with the results.
5. **Clip limits are stated.** Training skips clips longer than 30
   seconds (Whisper's input limit). Evaluation manifests state their clip
   count and total duration — both are printed by `prepare_data.py` and
   stored per file in the evaluation output.

---

## 3. Baseline-first protocol

Fine-tuning is only an improvement if it beats the model it started
from, measured fairly. The order of operations is fixed:

1. **Build the evaluation manifests first** (corpus test split and the
   separate benchmark manifests), and do not change them afterwards.
2. **Record the stock model's scores before any fine-tuning**, with
   `echomind/evaluate_asr.py`: stock Whisper Large-v3 on the GPU PC, on
   the benchmark manifest, with the exact command saved. Keep the
   resulting `outputs/eval_results.json` as the baseline record (the
   FINETUNING_GUIDE covers this as the "before" score).
3. **Fine-tune** using `scripts/train_lora.py` on the training manifest
   only.
4. **Re-run the identical evaluation** — same manifest, same evaluator,
   same device and compute type — using the merged, CTranslate2-converted
   model. The only thing that changes between the two runs is the model.
5. **Report both rows together** (Section 4), including the hardware and
   settings. A fine-tuned score without its baseline beside it is not
   reported.

A quick CPU smoke run (small model, e.g. `base`, small clip limit) is
used to check the pipeline works end to end. It is a sanity check, never
a reportable result, and is labelled as such wherever it appears.

---

## 4. Results table

Cells stay **TBD** until the corresponding run has actually been
executed under the protocol above. No number is estimated or carried over
from another paper's dataset. (One row is already filled: the CPU smoke
test, which is a pipeline check and is labelled as such.)

| Model | Dataset / manifest | Clips | WER | CER | RTF | Notes |
|---|---|---|---|---|---|---|
| Whisper Large-v3 (stock, to be run on GPU PC) | US-benchmark CS — `data/benchmark_manifests/test.jsonl` | TBD | TBD | TBD | TBD | Baseline. Record GPU model, device and compute type (float16) with the run |
| Whisper Large-v3 + LoRA (fine-tuned, CTranslate2) | US-benchmark CS — `data/benchmark_manifests/test.jsonl` | TBD | TBD | TBD | TBD | Same manifest and settings as the baseline row — the comparison row |
| Whisper Large-v3 (stock, to be run on GPU PC) | Corpus US-CS test split — `data/manifests/test.jsonl` | TBD | TBD | TBD | TBD | Secondary view; corpus transcripts are model-validated, so benchmark numbers take precedence |
| Whisper Large-v3 + LoRA (fine-tuned, CTranslate2) | Corpus US-CS test split — `data/manifests/test.jsonl` | TBD | TBD | TBD | TBD | Same manifest and settings as the row above |
| Whisper base (stock, CPU smoke test) | Smoke eval — val+test of a 134-clip sample (corpus poetry + benchmark interviews), 23 Oct 2026 | 13 | 80.7% | 65.7% | 0.73 | Sanity check of the evaluation pipeline only — not a reportable result. Full detail in `docs/BASELINE_SMOKE_TEST.md` |

Further rows are added for: DER (diarization, on the EchoMind Meeting
Benchmark), EER / FAR / FRR (speaker identification trials), and
precision / recall (extraction, on human-labelled meetings) once those
datasets exist.

---

## 5. Error analysis plan

A single WER number hides *where* the model fails. After each evaluation
run, errors are analysed in this order:

1. **Code-switched vs Urdu-only lines, scored separately.** This is the
   project's central question, so it gets its own numbers. Two cuts are
   used:
   - **By subset:** the same model is scored on the US-benchmark-CS
     manifest (code-switched) and the US-benchmark-Std manifest
     (Urdu-only), built with the same `--benchmark` download and manifest
     steps.
   - **Within code-switched clips:** the worst-scoring clips are
     inspected to see whether the English words, the Urdu words, or the
     switch points themselves are being lost.
2. **Worst files first.** `evaluate_asr.py` stores per-file WER / CER, so
   the highest-error clips are reviewed directly: listen to the audio,
   read the reference and the hypothesis side by side, and classify each
   failure (English word dropped or translated, Urdu spelling variant,
   number or name wrong, overlapping speech, noisy audio, reference
   transcript itself wrong).
3. **Reference errors are counted, not hidden.** Corpus transcripts are
   model-generated; where the "reference" is wrong and the system is
   right, that is recorded as a data error. This is one reason final
   claims rest on the hand-checked benchmark.
4. **Findings feed the next iteration.** Each error class maps to an
   action: English words lost → more code-switched training data;
   spelling variants dominating CER–WER gap → review normalisation;
   overlap failures → diarization and segmentation work, not ASR work.
   The classification counts are kept with the run's results so the next
   fine-tuning round can show whether the targeted fix worked.
