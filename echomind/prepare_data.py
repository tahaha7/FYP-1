import argparse
import json
import random
from pathlib import Path

import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def find_audio(cat_dir, clip_name):
    p = cat_dir / "audio" / clip_name
    if p.exists():
        return p
    # benchmark files use .WAV, corpus uses .wav, case can differ
    for variant in (clip_name.lower(), clip_name.upper()):
        alt = cat_dir / "audio" / variant
        if alt.exists():
            return alt
    return p


def read_rows(data_dir):
    rows = []
    # corpus files are *_final_transcription.jsonl, benchmark uses clean_transcription.jsonl
    jsonl_files = sorted(Path(data_dir).rglob("*_final_transcription.jsonl"))
    jsonl_files += sorted(Path(data_dir).rglob("clean_transcription.jsonl"))
    for jf in jsonl_files:
        cat_dir = jf.parent
        with open(jf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                audio_path = find_audio(cat_dir, item.get("Audio_Clip", ""))
                rows.append({
                    "audio_path": str(audio_path),
                    # benchmark rows also have ground_truth, prefer that one
                    "transcription": (item.get("ground_truth") or item.get("Transcription") or "").strip(),
                    "speaker_id": item.get("Speaker_id", "unknown"),
                    "category": item.get("Audio_category", cat_dir.name),
                    "duration": float(item.get("Duration_seconds") or 0),
                    "confidence": item.get("Confidence_score"),
                    "_exists": audio_path.exists(),
                })
    return rows


def filter_rows(rows, min_dur, max_dur, min_conf):
    kept = []
    for r in rows:
        if not r["_exists"]:
            continue
        if not r["transcription"]:
            continue
        if r["duration"] < min_dur or r["duration"] > max_dur:
            continue
        if min_conf is not None and r["confidence"] is not None:
            if float(r["confidence"]) < min_conf:
                continue
        r.pop("_exists")
        kept.append(r)
    return kept


def split_rows(rows, seed, speaker_aware):
    rng = random.Random(seed)
    rows = rows[:]
    if speaker_aware:
        # keep speakers separate so test speakers are unseen
        speakers = list({r["speaker_id"] for r in rows})
        rng.shuffle(speakers)
        n = len(speakers)
        test_sp = set(speakers[:max(1, int(n * 0.05))])
        val_sp = set(speakers[max(1, int(n * 0.05)):max(2, int(n * 0.10))])
        train = [r for r in rows if r["speaker_id"] not in test_sp | val_sp]
        val = [r for r in rows if r["speaker_id"] in val_sp]
        test = [r for r in rows if r["speaker_id"] in test_sp]
    else:
        rng.shuffle(rows)
        n = len(rows)
        n_test = int(n * 0.05)
        n_val = int(n * 0.05)
        test = rows[:n_test]
        val = rows[n_test:n_test + n_val]
        train = rows[n_test + n_val:]
    return train, val, test


def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stats(rows):
    hours = sum(r["duration"] for r in rows) / 3600
    return f"{len(rows)} clips, {hours:.2f} hours"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speaker-aware", action="store_true")
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=60.0)
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="e.g. 0.8 to drop low confidence rows")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["dataset"]["local_dir"]
    out_dir = Path(args.out_dir or cfg["dataset"]["manifest_dir"])

    rows = read_rows(data_dir)
    print(f"found {len(rows)} rows in transcription files")
    rows = filter_rows(rows, args.min_duration, args.max_duration, args.min_confidence)
    print(f"after filtering: {len(rows)} rows")

    if not rows:
        print("nothing left - check the data dir, did the download finish?")
        return

    train, val, test = split_rows(rows, args.seed, args.speaker_aware)
    write_jsonl(train, out_dir / "train.jsonl")
    write_jsonl(val, out_dir / "val.jsonl")
    write_jsonl(test, out_dir / "test.jsonl")

    print(f"train: {stats(train)}")
    print(f"val:   {stats(val)}")
    print(f"test:  {stats(test)}")
    print(f"manifests written to {out_dir}")


if __name__ == "__main__":
    main()
