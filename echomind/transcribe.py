import argparse
import json

from faster_whisper import WhisperModel

_model = None
_model_key = None


def get_model(model_size="base", device="cpu", compute_type="int8"):
    global _model, _model_key
    # reuse the model, loading it again and again is slow. device and
    # compute type are part of the key, otherwise a cpu int8 model gets
    # reused when a gpu float16 run was asked for
    key = (model_size, device, compute_type)
    if _model is None or _model_key != key:
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _model_key = key
    return _model


def transcribe_file(audio_path, model_size="base", device="cpu", compute_type="int8",
                    language=None):
    # language=None lets whisper detect it. the fine-tuned model is trained
    # with the "ur" token, so pass language="ur" for it (urdu-english
    # code-switching is what that token learned on UrduSpeech)
    model = get_model(model_size, device, compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=True,          # silero vad, cuts silence
        word_timestamps=True,
        language=language,
    )

    out_segments = []
    full_text = []
    for seg in segments:
        words = []
        for w in (seg.words or []):
            words.append({
                "start": w.start, "end": w.end,
                "word": w.word, "probability": w.probability,
            })
        out_segments.append({
            "start": seg.start, "end": seg.end,
            "text": seg.text.strip(), "words": words,
        })
        full_text.append(seg.text.strip())

    return {
        "audio_path": str(audio_path),
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "text": " ".join(full_text).strip(),
        "segments": out_segments,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model-size", default="base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--language", default=None,
                    help="force a language token, e.g. ur (default: auto-detect)")
    ap.add_argument("--out", default=None, help="write json to this file")
    args = ap.parse_args()

    result = transcribe_file(args.audio, args.model_size, args.device,
                             args.compute_type, language=args.language)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
