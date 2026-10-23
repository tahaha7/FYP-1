# tests for evaluation scoring and data preparation - the parts that decide
# whether a reported WER can be trusted. no gpu or model download needed.
# run with: python -m pytest tests/ -q

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from echomind import prepare_data
from echomind.evaluate_asr import corpus_scores, score_pair
from echomind.text_norm import normalize_text

ROOT = Path(__file__).resolve().parent.parent


# ---------- text normalisation ----------

def test_normalize_unifies_arabic_and_urdu_letters():
    # same words, typed with arabic yeh / kaf / heh vs urdu forms
    arabic_forms = "كيا وه"
    urdu_forms = "کیا وہ"
    assert normalize_text(arabic_forms) == normalize_text(urdu_forms)


def test_normalize_drops_diacritics_and_urdu_punctuation():
    assert normalize_text("کَیا؟") == "کیا"
    assert normalize_text("بات۔ بات،") == \
        "بات بات"


def test_normalize_keeps_english_words_and_maps_urdu_digits():
    assert normalize_text("Meeting ۵ baje, OK!") == "meeting 5 baje ok"


def test_normalize_handles_empty():
    assert normalize_text(None) == ""
    assert normalize_text("  ‌ ") == ""


# ---------- WER / CER scoring ----------

def test_score_pair_counts_errors():
    s = score_pair("a b c d", "a x c")
    assert s["word_errors"] == 2      # one substitution, one deletion
    assert s["ref_words"] == 4
    assert s["wer"] == 0.5


def test_corpus_wer_weights_by_words_not_files():
    short_bad = score_pair("a b", "x y")                      # 2/2 wrong
    long_good = score_pair(" ".join("w%d" % i for i in range(18)),
                           " ".join("w%d" % i for i in range(18)))  # 0/18
    scores = corpus_scores([short_bad, long_good])
    assert scores["wer"] == pytest.approx(2 / 20)     # corpus: 10%
    assert scores["wer_macro"] == pytest.approx(0.5)  # per-file mean: 50%


def test_corpus_scores_empty():
    assert corpus_scores([])["wer"] is None


# ---------- splits ----------

def rows_for(speakers, per_speaker=3):
    return [{"audio_path": f"{sp}_{i}.wav", "speaker_id": sp, "duration": 1.0}
            for sp in speakers for i in range(per_speaker)]


def test_split_is_same_under_different_hash_seeds():
    # the old code shuffled list(set(...)), whose order depends on
    # PYTHONHASHSEED, so the same seed gave different splits per run
    code = (
        "import json; from echomind.prepare_data import split_rows; "
        "rows=[{'speaker_id':'S%d'%i,'audio_path':str(i)} for i in range(40)]; "
        "tr,va,te=split_rows(rows,42,True); "
        "print(json.dumps(sorted(r['speaker_id'] for r in te)))"
    )
    outputs = set()
    for hash_seed in ("1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed)
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                             capture_output=True, text=True, check=True)
        outputs.add(out.stdout.strip())
    assert len(outputs) == 1


def test_subset_of_reads_corpus_and_benchmark_paths():
    assert prepare_data.subset_of("data/raw/corpus/US-CS/short/drama") == "CS"
    assert prepare_data.subset_of(
        "data/benchmark_raw/benchmark/US-benchmark-Std/short/NEWS") == "Std"
    assert prepare_data.subset_of("somewhere/else") == "unknown"


# ---------- prepare_data main: benchmark stays test-only ----------

def write_category(cat_dir, jsonl_name, n, benchmark=False):
    (cat_dir / "audio").mkdir(parents=True)
    with open(cat_dir / jsonl_name, "w", encoding="utf-8") as f:
        for i in range(n):
            name = f"clip{i}.wav"
            (cat_dir / "audio" / name).write_bytes(b"")
            row = {"Audio_Clip": name, "Transcription": f"text {i}",
                   "Speaker_id": f"spk{i}", "Duration_seconds": 2.0}
            if benchmark:
                row["ground_truth"] = f"gold {i}"
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def project(tmp_path, monkeypatch):
    cfg = {"dataset": {
        "local_dir": str(tmp_path / "raw"),
        "manifest_dir": str(tmp_path / "manifests"),
        "benchmark_dir": str(tmp_path / "benchmark_raw"),
        "benchmark_manifest_dir": str(tmp_path / "benchmark_manifests"),
    }}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    write_category(tmp_path / "raw/corpus/US-CS/short/interview",
                   "interview_final_transcription.jsonl", 40)
    write_category(tmp_path / "benchmark_raw/benchmark/US-benchmark-CS/short/INTERVIEWS",
                   "clean_transcription.jsonl", 10, benchmark=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run_prepare(*args):
    old = sys.argv
    sys.argv = ["prepare_data", "--config", "config.yaml", *args]
    try:
        prepare_data.main()
    finally:
        sys.argv = old


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l]


def test_benchmark_manifest_is_all_test_rows(project):
    run_prepare("--benchmark")
    out = project / "benchmark_manifests"
    rows = read_jsonl(out / "test.jsonl")
    assert len(rows) == 10                       # nothing split off
    assert not (out / "train.jsonl").exists()
    assert all(r["source"] == "benchmark" and r["subset"] == "CS" for r in rows)
    assert rows[0]["transcription"] == "gold 0"  # ground_truth preferred


def test_corpus_manifests_are_speaker_aware_by_default(project):
    run_prepare()
    splits = {name: read_jsonl(project / "manifests" / f"{name}.jsonl")
              for name in ("train", "val", "test")}
    speakers = {name: {r["speaker_id"] for r in rows} for name, rows in splits.items()}
    assert speakers["train"].isdisjoint(speakers["test"])
    assert speakers["train"].isdisjoint(speakers["val"])
    assert all(r["source"] == "corpus" for rows in splits.values() for r in rows)


def test_mixed_corpus_and_benchmark_is_refused(project):
    write_category(project / "raw/benchmark/US-benchmark-CS/short/NEWS",
                   "clean_transcription.jsonl", 3, benchmark=True)
    with pytest.raises(SystemExit, match="test-only"):
        run_prepare()
    assert not (project / "manifests").exists()


def test_benchmark_cannot_be_written_to_training_manifest_dir(project):
    with pytest.raises(SystemExit):
        run_prepare("--benchmark", "--out-dir", str(project / "manifests"))
