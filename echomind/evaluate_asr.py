import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import jiwer

from echomind.run_info import run_info
from echomind.text_norm import normalize_text

# kept as a name so old imports still work
normalize = normalize_text


def load_manifest(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def score_pair(ref, hyp):
    # ref and hyp are already normalised. returns raw error counts so the
    # corpus score can be summed over files instead of averaging percentages
    w = jiwer.process_words(ref, hyp)
    c = jiwer.process_characters(ref, hyp)
    ref_words = w.hits + w.substitutions + w.deletions
    ref_chars = c.hits + c.substitutions + c.deletions
    return {
        "word_errors": w.substitutions + w.deletions + w.insertions,
        "ref_words": ref_words,
        "char_errors": c.substitutions + c.deletions + c.insertions,
        "ref_chars": ref_chars,
        "wer": (w.substitutions + w.deletions + w.insertions) / ref_words,
        "cer": (c.substitutions + c.deletions + c.insertions) / ref_chars,
    }


def corpus_scores(items):
    # corpus wer = all word errors / all reference words. this is the
    # standard number to report. the per-file average (macro) is kept only
    # for comparison - it lets a 3 word clip count as much as a 40 word one
    if not items:
        return {"files": 0, "wer": None, "cer": None,
                "wer_macro": None, "cer_macro": None}
    word_errors = sum(i["word_errors"] for i in items)
    ref_words = sum(i["ref_words"] for i in items)
    char_errors = sum(i["char_errors"] for i in items)
    ref_chars = sum(i["ref_chars"] for i in items)
    return {
        "files": len(items),
        "ref_words": ref_words,
        "wer": word_errors / ref_words,
        "cer": char_errors / ref_chars,
        "wer_macro": sum(i["wer"] for i in items) / len(items),
        "cer_macro": sum(i["cer"] for i in items) / len(items),
    }


def grouped_scores(items, key):
    groups = defaultdict(list)
    for i in items:
        groups[i.get(key) or "unknown"].append(i)
    return {name: corpus_scores(rows) for name, rows in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/test.jsonl")
    ap.add_argument("--model-size", default="base",
                    help="faster-whisper size, or path to a converted ct2 model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--language", default=None,
                    help="force a language token, e.g. ur. default: auto-detect. "
                         "run both for the baseline, the fine-tuned model is "
                         "trained with ur")
    ap.add_argument("--out", default="outputs/eval_results.json")
    args = ap.parse_args()

    # imported here so the scoring functions above can be used and tested
    # without faster-whisper installed
    from echomind.transcribe import transcribe_file

    rows = load_manifest(args.manifest, args.limit)
    print(f"{len(rows)} rows in manifest")

    results = []
    missing_audio = 0
    empty_reference = 0
    for i, row in enumerate(rows, 1):
        audio_path = row.get("audio_path", "")
        if not Path(audio_path).exists():
            missing_audio += 1
            continue
        ref = normalize_text(row.get("transcription", ""))
        if not ref:
            empty_reference += 1
            continue

        t0 = time.time()
        out = transcribe_file(audio_path, args.model_size, args.device,
                              args.compute_type, language=args.language)
        took = time.time() - t0

        hyp = normalize_text(out["text"])
        duration = float(row.get("duration") or out.get("duration") or 0)

        item = {
            "audio_path": audio_path,
            "subset": row.get("subset"),
            "category": row.get("category"),
            "reference": row.get("transcription", ""),
            "hypothesis": out["text"],
            "detected_language": out.get("language"),
            "duration": duration,
            "seconds": took,
            "rtf": (took / duration) if duration else None,
            **score_pair(ref, hyp),
        }
        results.append(item)
        print(f"[{i}] wer={item['wer']:.3f} cer={item['cer']:.3f} "
              f"rtf={item['rtf']}  {Path(audio_path).name}")

    overall = corpus_scores(results)
    total_audio = sum(r["duration"] for r in results)
    total_seconds = sum(r["seconds"] for r in results)
    overall["rtf"] = (total_seconds / total_audio) if total_audio else None
    overall["audio_hours"] = total_audio / 3600

    print(f"\nfiles scored: {len(results)}, skipped: {missing_audio} missing "
          f"audio, {empty_reference} empty reference")
    print(f"corpus WER: {overall['wer']}  CER: {overall['cer']}  "
          f"RTF: {overall['rtf']}")
    print(f"(per-file average, not for reporting: WER {overall['wer_macro']} "
          f"CER {overall['cer_macro']})")

    report = {
        "run": run_info({
            "model_size": args.model_size,
            "manifest": args.manifest,
            "limit": args.limit,
            "device": args.device,
            "compute_type": args.compute_type,
            "language": args.language,
        }, manifest_path=args.manifest),
        "files_scored": len(results),
        "files_skipped_missing_audio": missing_audio,
        "files_skipped_empty_reference": empty_reference,
        "overall": overall,
        "by_subset": grouped_scores(results, "subset"),
        "by_category": grouped_scores(results, "category"),
        "per_file": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
