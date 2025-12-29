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
CUDA_VISIBLE_DEVICES=5 \
python train/train_helm_v2/train_helm.py \
  --model_name_or_path "/ckpt/Qwen2.5-VL-7B-Instruct" \
  --train_jsonl "/data/ghkim/helm_data/wipe_the_window/jsonl_v2/merged/wipe_the_window/all_train.jsonl" \
  --val_jsonl   "/data/ghkim/helm_data/wipe_the_window/jsonl_v2/merged/wipe_the_window/all_val.jsonl" \
  --frames_root "/data/ghkim/helm_data/wipe_the_window" \
  --camera table \
  --num_images 1 \
  --output_dir "/result/ghkim/HLP_HeLM_v2_wipe_the_window_1229" \
  --use_qlora 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --eval_steps 50 \
  --save_steps 200 \
  --max_steps 2000 \
  --wandb_project RefMe \
  --wandb_run_name HLP_HeLM_v2_wipe_the_window_1229
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
    model.config.use_cache = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)

    # can be file or folder
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)

    # v2 needs this for update rows (reconstruct table image at t_event)
    ap.add_argument("--frames_root", type=str, default=None, help="out_root that contains frames_1hz/ ...")
    ap.add_argument("--camera", type=str, default="table")

    ap.add_argument("--num_images", type=int, default=1)

    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--use_qlora", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--warmup_steps", type=int, default=50)

    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--wandb_project", type=str, default="helm")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    try:
        import wandb  # type: ignore
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    except Exception as e:
        print(f"[WARN] wandb.init failed or wandb not installed: {e}")

    train_ds = HelmJsonlDataset(
        jsonl_path=args.train_jsonl,
        model_name_or_path=args.model_name_or_path,
        num_image=args.num_images,
        camera=args.camera,
        frames_root=args.frames_root,
        require_images_for_update=True,  # keep batching VLM-safe
    )
    val_ds = HelmJsonlDataset(
        jsonl_path=args.val_jsonl,
        model_name_or_path=args.model_name_or_path,
        num_image=args.num_images,
        camera=args.camera,
        frames_root=args.frames_root,
        require_images_for_update=True,
    )

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

        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,

        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_num_workers=args.num_workers,

        remove_unused_columns=False,

        report_to=["wandb"],
        log_level="info",
        logging_first_step=True,
        max_grad_norm=1.0,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[WandbStaticStatsCallback()],
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()