from __future__ import annotations

import argparse
import os

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from helm_dataset import HelmJsonlDataset, HelmDataCollator

"""
CUDA_VISIBLE_DEVICES=7 python train/train_helm/train_helm.py \
  --model_name_or_path "/ckpt/Qwen2.5-VL-7B-Instruct" \
  --train_jsonl "/data/ghkim/helm_data/press_the_button_N_times_ep60/jsonl/merged/press_1+2+3/all_train.jsonl" \
  --val_jsonl   "/data/ghkim/helm_data/press_the_button_N_times_ep60/jsonl/merged/press_1+2+3/all_val.jsonl" \
  --output_dir "/result/ghkim/HLP_HeLM_press_1+2+3_1223" \
  --wandb_project "RefMe" \
  --wandb_run_name "HLP_HeLM_press_1+2+3_1223" \
  --use_qlora 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --eval_steps 50 \
  --save_steps 300 \
  --max_steps 1500
"""

def count_parameters(model) -> tuple[int, int, float]:
    all_params = 0
    trainable_params = 0
    for p in model.parameters():
        n = p.numel()
        all_params += n
        if p.requires_grad:
            trainable_params += n
    pct = 100.0 * trainable_params / max(1, all_params)
    return trainable_params, all_params, pct


class WandbStaticStatsCallback(TrainerCallback):
    """Logs trainable/all params once at train begin (to W&B via Trainer.log)."""

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)
        if trainer is None or model is None:
            return
        trainable_params, all_params, pct = count_parameters(model)
        trainer.log({
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": pct,
        })


def build_model(model_name_or_path: str, use_qlora: bool):
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.config.use_cache = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--use_qlora", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=4)

    # W&B
    ap.add_argument("--wandb_project", type=str, default="helm")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # W&B env (Trainer가 자동으로 wandb.init 함)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # Ensure deterministic W&B naming (avoid HF fallback to output_dir as project/name).
    try:
        import wandb  # type: ignore
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    except Exception as e:
        print(f"[WARN] wandb.init failed or wandb not installed: {e}")

    train_ds = HelmJsonlDataset(args.train_jsonl, args.model_name_or_path)
    val_ds = HelmJsonlDataset(args.val_jsonl, args.model_name_or_path)
    collator = HelmDataCollator(pad_id=train_ds.processor.tokenizer.pad_token_id)

    model = build_model(args.model_name_or_path, use_qlora=bool(args.use_qlora))

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=args.wandb_run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,

        # ✅ logging/eval/save
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,

        # ✅ mixed precision / perf
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=args.num_workers,

        # ✅ important for VLM tensors
        remove_unused_columns=False,

        # ✅ W&B
        report_to=["wandb"],

        # ✅ makes Trainer log learning_rate / grad_norm more reliably
        log_level="info",
        logging_first_step=True,
        max_grad_norm=1.0,

        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,   # ✅ val loss 계산
        data_collator=collator,
        callbacks=[WandbStaticStatsCallback()],  # ✅ trainable/all params 로깅
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()