#!/usr/bin/env python
"""
Seq2seq refinement finetune (ProtT5 encoder+decoder) on (noisy -> clean) pairs.

Inputs (TSV with 3 columns):  ctrl \t noisy \t target
Produced by: scripts/build_refine_pairs_all.py

Training input  = ctrl + " " + space_separated(noisy)
Training target = space_separated(target)

Supports:
- Full finetune of ALL ~2.84B params (use: --no-use-lora) with 8-bit optimizer
- LoRA finetune (default --use-lora) on attention proj layers

Version-agnostic: auto-adjusts TrainingArguments keys for older transformers.
"""

import os, inspect, argparse
import pandas as pd
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from peft import LoraConfig, get_peft_model
import torch


# ------------------------ helpers ------------------------

def space_tokens(s: str) -> str:
    s = s.replace(" ", "").upper()
    return " ".join(list(s))

def load_refine_tsv(path: str) -> Dataset:
    df = pd.read_csv(path, sep="\t", header=None, names=["ctrl","noisy","tgt"])
    df["inp"] = df["ctrl"].astype(str) + " " + df["noisy"].astype(str).apply(space_tokens)
    df["tgt"] = df["tgt"].astype(str).apply(space_tokens)
    return Dataset.from_pandas(df[["inp","tgt"]])

def build_training_args(save_dir, epochs, bsz, grad_accum, lr, max_workers):
    ta_sig = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = {
        "output_dir": save_dir,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": bsz,
        "per_device_eval_batch_size": max(1, bsz),
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "fp16": False,              # set below based on GPU
        "bf16": False,              # set below based on GPU
        "gradient_checkpointing": True,
        "dataloader_num_workers": max_workers,
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "report_to": "none",
    }

    # Prefer bf16 on Ampere+ (A6000), else fp16
    supports_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    ta_kwargs["bf16"] = supports_bf16
    ta_kwargs["fp16"] = False if supports_bf16 else True

    # Enable TF32 where possible
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if "tf32" in ta_sig:
        ta_kwargs["tf32"] = True

    # Fall back for older transformers without eval/save strategies
    if "evaluation_strategy" not in ta_sig:
        ta_kwargs.pop("evaluation_strategy", None)
        if "save_strategy" in ta_sig:
            ta_kwargs["save_strategy"] = "steps"
            ta_kwargs["save_steps"] = 2000
        elif "save_steps" in ta_sig:
            ta_kwargs["save_steps"] = 2000
        ta_kwargs["load_best_model_at_end"] = False
        if "logging_steps" not in ta_sig and "logging_first_step" in ta_sig:
            ta_kwargs.pop("logging_steps", None)

    # Filter unknown kwargs dynamically
    ta_final = {k: v for k, v in ta_kwargs.items() if k in ta_sig}
    return ta_sig, ta_final


# ------------------------ main ------------------------

def main():
    # argparse (clean one-time definition of --use_lora)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Rostlab/prot_t5_xl_uniref50")
    ap.add_argument("--train_tsv", default="data/train_refine.tsv")
    ap.add_argument("--val_tsv",   default="data/val_refine.tsv")
    ap.add_argument("--save_dir",  default="runs/protT5_refiner_lora")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_in_len", type=int, default=256)
    ap.add_argument("--max_out_len", type=int, default=256)
    ap.add_argument(
        "--use_lora",
        action=argparse.BooleanOptionalAction,  # supports --use-lora / --no-use-lora
        default=True,
        help="Use LoRA adapters; pass --no-use-lora for full finetuning."
    )
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # tokenizer + model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False, legacy=False)

    # Prefer bf16 on Ampere+ else fp16; use dtype if supported, else fallback load
    want_dtype = None
    if torch.cuda.is_available():
        want_dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name, use_safetensors=True, low_cpu_mem_usage=True, dtype=want_dtype
        )
    except TypeError:
        # older transformers don't accept `dtype`
        model = T5ForConditionalGeneration.from_pretrained(
            args.model_name, use_safetensors=True, low_cpu_mem_usage=True
        )

    # Save VRAM
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA (encoder+decoder attn) OR full finetune
    if args.use_lora:
        peft_cfg = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q","k","v","o"],
            inference_mode=False,
        )
        model = get_peft_model(model, peft_cfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params with LoRA: {trainable:,} / {total:,}")
    else:
        for p in model.parameters():
            p.requires_grad = True
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full finetune (no LoRA): trainable params {trainable:,} / {total:,}")

    # datasets
    train_ds = load_refine_tsv(args.train_tsv)
    val_ds   = load_refine_tsv(args.val_tsv)

    # tokenize with text_target API
    def preprocess(examples):
        model_inputs = tokenizer(
            examples["inp"],
            max_length=args.max_in_len,
            padding="max_length",
            truncation=True,
        )
        labels = tokenizer(
            text_target=examples["tgt"],
            max_length=args.max_out_len,
            padding="max_length",
            truncation=True,
        )["input_ids"]
        pad = tokenizer.pad_token_id
        labels = [[(tok if tok != pad else -100) for tok in seq] for seq in labels]
        model_inputs["labels"] = labels
        return model_inputs

    train_tok = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    # collator
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # training args (version-agnostic)
    ta_sig, ta_final = build_training_args(
        save_dir=args.save_dir,
        epochs=args.epochs,
        bsz=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_workers=args.num_workers,
    )

    # Choose optimizer: 8-bit for FULL FT if available; torch AdamW for LoRA
    if "optim" in ta_sig:
        ta_final["optim"] = "adamw_torch" if args.use_lora else "adamw_bnb_8bit"

    # If eval strategy not supported, ensure load_best_model_at_end is off
    if "evaluation_strategy" not in ta_sig and "load_best_model_at_end" in ta_final:
        ta_final["load_best_model_at_end"] = False

    args_train = TrainingArguments(**ta_final)

    # trainer
    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=train_tok,
        eval_dataset=val_tok if ("evaluation_strategy" in ta_sig or "eval_steps" in ta_sig) else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("Saved model to", args.save_dir)


if __name__ == "__main__":
    main()
