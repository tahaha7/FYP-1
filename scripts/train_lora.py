import argparse
import json
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

import jiwer

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

    def prepare(batch):
        # load audio as 16khz mono, whisper needs exactly that
        audio, _ = librosa.load(batch["audio_path"], sr=16000, mono=True)
        batch["input_features"] = feature_extractor(
            audio, sampling_rate=16000).input_features[0]
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    train_ds = Dataset.from_list(raw["train"])
    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names,
                            desc="train features")
    eval_ds = None
    if raw["val"]:
        eval_ds = Dataset.from_list(raw["val"])
        eval_ds = eval_ds.map(prepare, remove_columns=eval_ds.column_names,
                              desc="val features")

    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        # standard collator from the hf whisper fine-tuning blog
        processor: object
        decoder_start_token_id: int

        def __call__(self, features):
            input_features = [
                {"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(
                label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100)

            # drop the starting token, the model adds it itself
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    print(f"loading base model {args.base_model} ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, load_in_8bit=False, device_map="auto")
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
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = jiwer.wer(
            [s.strip().lower() for s in label_str],
            [s.strip().lower() for s in pred_str],
        )
        return {"wer": wer * 100}

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
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
        fp16=not use_bf16,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="wer" if eval_ds is not None else None,
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

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
