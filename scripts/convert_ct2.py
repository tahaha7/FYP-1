import argparse
import subprocess
import sys

# converts a full hf whisper model folder into the faster-whisper (ct2) format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/whisper-merged",
                    help="full hf model dir (run merge_lora.py first, "
                         "an adapter folder alone will not work)")
    ap.add_argument("--out", default="models/whisper-lora-ct2")
    ap.add_argument("--quantization", default="float16",
                    choices=["float16", "float32", "int8", "int8_float16"])
    args = ap.parse_args()

    cmd = [
        "ct2-transformers-converter",
        "--model", args.model,
        "--output_dir", args.out,
        "--quantization", args.quantization,
        "--copy_files", "tokenizer.json", "preprocessor_config.json",
    ]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("converter failed - check the error above")
        sys.exit(result.returncode)

    print(f"\ndone. converted model is in {args.out}")
    print("use it like this:")
    print(f'  python -m echomind.transcribe meeting.wav --model-size {args.out} --device cuda --compute-type float16')
    print(f'  python -m echomind.evaluate_asr --model-size {args.out} --device cuda --compute-type float16')
    print(f'  python -m echomind.orchestrator meeting.wav --model-size {args.out} --device cuda --compute-type float16')
    print("or in python:")
    print(f'  from faster_whisper import WhisperModel')
    print(f'  model = WhisperModel("{args.out}", device="cuda", compute_type="float16")')


if __name__ == "__main__":
    main()
