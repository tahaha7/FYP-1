import argparse

import torch
from peft import PeftModel
from transformers import AutoProcessor, WhisperForConditionalGeneration

# the ct2 converter needs a full model, not just a lora adapter
# so this merges the adapter back into the base model first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="openai/whisper-large-v3")
    ap.add_argument("--adapter", default="models/whisper-lora")
    ap.add_argument("--out", default="models/whisper-merged")
    args = ap.parse_args()

    print(f"loading base model {args.base_model}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.float16)

    print(f"loading adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("merging...")
    model = model.merge_and_unload()

    model.save_pretrained(args.out)
    processor = AutoProcessor.from_pretrained(args.adapter)
    processor.save_pretrained(args.out)
    print(f"merged model saved to {args.out}")
    print(f"next: python scripts/convert_ct2.py --model {args.out}")


if __name__ == "__main__":
    main()
