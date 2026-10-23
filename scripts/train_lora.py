import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model

# scripts/ is not a package, so make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echomind.evaluate_asr import corpus_scores, score_pair  # noqa: E402
from echomind.run_info import run_info  # noqa: E402
from echomind.text_norm import normalize_text  # noqa: E402

# fine-tunes whisper with lora on the urduspeech manifests
# run this on a gpu pc, not on cpu - it will take forever there


def load_manifests(manifest_dir, max_duration=30.0):
    data = {"train": [], "val": []}
    files = {"train": "train.jsonl", "val": "val.jsonl"}
    for split, fname in files.items():
        path = Path(manifest_dir) / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    # whisper takes max 30s, skip longer clips
                    if float(row.get("duration") or 0) > max_duration:
                        continue
                    data[split].append({
                        "audio_path": row["audio_path"],
                        "transcription": row["transcription"],
                    })
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--base-model", default="openai/whisper-large-v3")
    ap.add_argument("--output-dir", default="models/whisper-lora")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--language", default="ur",
                    help="language token for whisper. ur fits because most of "
                         "the data is urdu, english words still come out fine")
    ap.add_argument("--max-duration", type=float, default=30.0,
                    help="whisper only takes 30s at a time, longer clips are skipped in prepare below")
    args = ap.parse_args()

    print("loading manifests...")
    raw = load_manifests(args.manifest_dir, args.max_duration)
    if not raw["train"]:
        raise SystemExit("no train rows found - run prepare_data.py first")
    print(f"train: {len(raw['train'])} clips, val: {len(raw['val'])} clips")

    processor = AutoProcessor.from_pretrained(args.base_model)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # language + task tokens, standard whisper setup
    tokenizer.set_prefix_tokens(language=args.language, task="transcribe")

    # whisper's decoder takes at most 448 tokens, longer labels would crash
    max_label_tokens = 448

    def keep(row):
        return len(tokenizer(row["transcription"]).input_ids) <= max_label_tokens

    # only paths and text are stored. the log-mel features are computed in
    # the collator for each batch - precomputing them for every clip needs
    # about 1.5 MB per clip, around 100 GB for the full CS subset
    train_ds = Dataset.from_list(raw["train"]).filter(keep, desc="train labels")
    eval_ds = None
    if raw["val"]:
        eval_ds = Dataset.from_list(raw["val"]).filter(keep, desc="val labels")
    print(f"after label length check: train {len(train_ds)}, "
          f"val {len(eval_ds) if eval_ds is not None else 0}")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        # collator from the hf whisper fine-tuning blog, but it loads the
        # audio itself so features are never stored on disk
        processor: object
        decoder_start_token_id: int

        def __call__(self, features):
            input_features = []
            label_features = []
            for f in features:
                # 16khz mono, whisper needs exactly that
                audio, _ = librosa.load(f["audio_path"], sr=16000, mono=True)
                feats = self.processor.feature_extractor(
                    audio, sampling_rate=16000).input_features[0]
                input_features.append({"input_features": feats})
                label_features.append(
                    {"input_ids": self.processor.tokenizer(f["transcription"]).input_ids})

            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt")

            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)

            # drop the starting token, the model adds it itself
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    # bf16 halves memory (about 3 GB of weights instead of 6). older gpus
    # without bf16 load in fp32 - fp16 base weights make the grad scaler fail
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"loading base model {args.base_model} in {dtype} ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=dtype)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # lora - train only a small part of the model, rest stays frozen
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # both can be padded with -100, which the tokenizer cannot decode
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # same normaliser and corpus-level scoring as evaluate_asr.py, so the
        # validation wer here is comparable with the reported numbers
        items = []
        for ref, hyp in zip(label_str, pred_str):
            ref = normalize_text(ref)
            if ref:
                items.append(score_pair(ref, normalize_text(hyp)))
        scores = corpus_scores(items)
        return {"wer": (scores["wer"] or 0.0) * 100,
                "cer": (scores["cer"] or 0.0) * 100}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer" if eval_ds is not None else None,
        greater_is_better=False,
        # audio_path and transcription must reach the collator
        remove_unused_columns=False,
        # a peft-wrapped model hides its forward() signature, so the trainer
        # cannot find the labels by itself and would skip eval loss and wer
        label_names=["labels"],
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    info = run_info(vars(args), manifest_path=Path(args.manifest_dir) / "train.jsonl")
    info["train_clips"] = len(train_ds)
    info["val_clips"] = len(eval_ds) if eval_ds is not None else 0
    with open(Path(args.output_dir) / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("starting training - checkpoints are saved every epoch in the")
    print("output folder, so a crash does not lose everything.")
    trainer.train()

    # save the adapter + processor, this is what merge_lora.py needs
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"done. adapter saved to {args.output_dir}")
    print("next: python scripts/merge_lora.py")
    print("then: python scripts/convert_ct2.py")


if __name__ == "__main__":
    main()
