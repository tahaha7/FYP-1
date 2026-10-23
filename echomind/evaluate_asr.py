import argparse
import json
import re
import time
from pathlib import Path

import jiwer

from echomind.transcribe import transcribe_file

# urdu + english punctuation we dont want in the score
PUNCT = ".,!?;:'\"()[]{}-–—" + "؟،۔؛«»"


def normalize(text):
    text = (text or "").lower()
    for ch in PUNCT:
        text = text.replace(ch, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/test.jsonl")
    ap.add_argument("--model-size", default="base",
                    help="faster-whisper size, or path to a converted ct2 model")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--out", default="outputs/eval_results.json")
    args = ap.parse_args()

    rows = load_manifest(args.manifest, args.limit)
    print(f"{len(rows)} rows in manifest")

    results = []
    skipped = 0
    for i, row in enumerate(rows, 1):
        audio_path = row.get("audio_path", "")
        if not Path(audio_path).exists():
            skipped += 1
            continue

        t0 = time.time()
        out = transcribe_file(audio_path, args.model_size, args.device,
                              args.compute_type)
        took = time.time() - t0

        ref = normalize(row.get("transcription", ""))
        hyp = normalize(out["text"])
        duration = float(row.get("duration") or out.get("duration") or 0)

        # jiwer needs at least something in ref/hyp
        wer = jiwer.wer(ref, hyp) if ref and hyp else (0.0 if ref == hyp else 1.0)
        cer = jiwer.cer(ref, hyp) if ref and hyp else (0.0 if ref == hyp else 1.0)

        item = {
            "audio_path": audio_path,
            "reference": row.get("transcription", ""),
            "hypothesis": out["text"],
            "wer": wer,
            "cer": cer,
            "duration": duration,
            "rtf": (took / duration) if duration else None,
        }
        results.append(item)
        print(f"[{i}] wer={wer:.3f} cer={cer:.3f} rtf={item['rtf']}  {Path(audio_path).name}")

    if results:
        avg_wer = sum(r["wer"] for r in results) / len(results)
        avg_cer = sum(r["cer"] for r in results) / len(results)
        rtfs = [r["rtf"] for r in results if r["rtf"] is not None]
        avg_rtf = sum(rtfs) / len(rtfs) if rtfs else None
    else:
        avg_wer = avg_cer = avg_rtf = None

    print(f"\nfiles scored: {len(results)}, skipped (missing audio): {skipped}")
    print(f"overall WER: {avg_wer}  CER: {avg_cer}  RTF: {avg_rtf}")

    report = {
        "model_size": args.model_size,
        "manifest": args.manifest,
        "files_scored": len(results),
        "files_skipped": skipped,
        "wer": avg_wer,
        "cer": avg_cer,
        "rtf": avg_rtf,
        "per_file": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
