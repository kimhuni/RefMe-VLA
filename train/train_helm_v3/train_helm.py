# train_helm_v3.py
from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from train.train_helm_v3.helm_dataset import (
    HelmJsonlDatasetV3,
    V3DatasetConfig,
    DataCollatorForQwenVL,
    read_jsonl,
    count_labels,
)

"""
CUDA_VISIBLE_DEVICES=4 python train/train_helm_v3/train_helm.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/wipe_the_window/jsonl_v3/merged/wipe_the_window/all_train.jsonl \
  --val_jsonl /data/ghkim/helm_data/wipe_the_window/jsonl_v3/merged/wipe_the_window/all_val.jsonl \
  --num_images 1 \
  --output_dir /result/ghkim/HLP_HeLM_v3_wipe_the_window_2240_0101 \
  --batch_size 8 --n_detect_pos 2 --n_detect_neg 2 --n_update_intra 4 --n_update_transition 0 \
  --num_train_epochs 10 \
  --with_replacement True \
  --attn_impl sdpa \
  --eval_max_samples 40 \
  --wandb_project RefMe \
  --wandb_run_name HLP_HeLM_v3_wipe_the_window_2240_0101
  
CUDA_VISIBLE_DEVICES=5 python train/train_helm_v3/train_helm.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/wipe_the_window/jsonl_v3/merged/wipe_the_window/all_train.jsonl \
  --val_jsonl /data/ghkim/helm_data/wipe_the_window/jsonl_v3/merged/wipe_the_window/all_val.jsonl \
  --num_images 1 \
  --output_dir /result/ghkim/HLP_HeLM_v3_wipe_the_window_3320_0101 \
  --batch_size 8 --n_detect_pos 3 --n_detect_neg 3 --n_update_intra 2 --n_update_transition 0 \
  --num_train_epochs 10 \
  --with_replacement True \
  --attn_impl sdpa \
  --eval_max_samples 40 \
  --wandb_project RefMe \
  --wandb_run_name HLP_HeLM_v3_wipe_the_window_3320_0101
  
CUDA_VISIBLE_DEVICES=6 python train/train_helm_v3/train_helm.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/press_the_button_nolight/jsonl_v3/merged/all_train.jsonl \
  --val_jsonl /data/ghkim/helm_data/press_the_button_nolight/jsonl_v3/merged/all_val.jsonl \
  --num_images 1 \
  --output_dir /result/ghkim/HLP_HeLM_v3_press_N_3320_0101 \
  --batch_size 8 --n_detect_pos 3 --n_detect_neg 3 --n_update_intra 2 --n_update_transition 0 \
  --num_train_epochs 10 \
  --with_replacement True \
  --attn_impl sdpa \
  --eval_max_samples 40 \
  --wandb_project RefMe \
  --wandb_run_name HLP_HeLM_v3_press_N_3320_0101
"""

# ---------------------------
# Mixed Batch Sampler
# ---------------------------
class MixedBatchSampler:
    """
    Yield list[int] indices each step:
      per_batch = {"detect_pos":2, "detect_neg":2, "update_intra":4, "update_transition":0}
    """
    def __init__(
        self,
        pools: Dict[str, List[int]],
        per_batch: Dict[str, int],
        steps_per_epoch: int,
        seed: int = 0,
        with_replacement: bool = True,
        shuffle_within_batch: bool = True,
    ):
        self.pools = {k: list(v) for k, v in pools.items()}
        self.per_batch = dict(per_batch)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.with_replacement = bool(with_replacement)
        self.shuffle_within_batch = bool(shuffle_within_batch)

        # validate
        for k, n in self.per_batch.items():
            if n < 0:
                raise ValueError(f"per_batch[{k}] must be >=0")
            if n > 0 and len(self.pools.get(k, [])) == 0:
                raise ValueError(f"pool '{k}' is empty but per_batch[{k}]={n}")

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)

        if self.with_replacement:
            for _ in range(self.steps_per_epoch):
                batch: List[int] = []
                for k, n in self.per_batch.items():
                    if n == 0:
                        continue
                    batch.extend(rng.choices(self.pools[k], k=n))
                if self.shuffle_within_batch:
                    rng.shuffle(batch)
                yield batch
        else:
            # no replacement: shuffle pools and consume
            working = {k: list(v) for k, v in self.pools.items()}
            for k in working:
                rng.shuffle(working[k])

            for _ in range(self.steps_per_epoch):
                batch: List[int] = []
                for k, n in self.per_batch.items():
                    if n == 0:
                        continue
                    if len(working[k]) < n:
                        return
                    batch.extend(working[k][:n])
                    del working[k][:n]
                if self.shuffle_within_batch:
                    rng.shuffle(batch)
                yield batch


# ---------------------------
# Trainer override
# ---------------------------
class MixedBatchTrainer(Trainer):
    def __init__(self, *args, train_batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_batch_sampler = train_batch_sampler

    def get_train_dataloader(self):
        if self._train_batch_sampler is None:
            return super().get_train_dataloader()

        return DataLoader(
            self.train_dataset,
            batch_sampler=self._train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )


# ---------------------------
# misc utils
# ---------------------------
def count_parameters(model) -> tuple[int, int, float]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100.0 * trainable / max(total, 1)
    return trainable, total, pct


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)
        if trainer is None or model is None:
            return
        trainable, total, pct = count_parameters(model)
        trainer.log({
            "trainable_params": trainable,
            "all_params": total,
            "trainable_percentage": pct,
        })


def build_model(model_name_or_path: str, use_qlora: bool, bf16: bool, attn_impl: str):
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

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
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)

    ap.add_argument("--num_images", type=int, default=1, choices=[1, 2])
    ap.add_argument("--output_dir", type=str, required=True)

    # batch mixture knobs
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_detect_pos", type=int, default=2)
    ap.add_argument("--n_detect_neg", type=int, default=2)
    ap.add_argument("--n_update_intra", type=int, default=4)
    ap.add_argument("--n_update_transition", type=int, default=0)

    ap.add_argument("--with_replacement", type=bool, default=True)
    ap.add_argument("--sampler_seed", type=int, default=0)

    # epoch sizing
    ap.add_argument("--steps_per_epoch", type=int, default=0,
                    help="0 => auto: floor(total_train_rows / batch_size)")
    ap.add_argument("--num_train_epochs", type=float, default=3.0)

    # eval sampling
    ap.add_argument("--eval_max_samples", type=int, default=0,
                    help="0 => use full val; else randomly sample this many")
    ap.add_argument("--eval_seed", type=int, default=123)

    # training args
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)

    ap.add_argument("--bf16", type=bool, default=True)
    ap.add_argument("--use_qlora", type=bool, default=True)

    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    # wandb
    ap.add_argument("--wandb_project", type=str, default="RefMe")
    ap.add_argument("--wandb_run_name", type=str, default="")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # -------- validate batch composition --------
    per_batch = {
        "detect_pos": args.n_detect_pos,
        "detect_neg": args.n_detect_neg,
        "update_intra": args.n_update_intra,
        "update_transition": args.n_update_transition,
    }
    mix_sum = sum(per_batch.values())
    if mix_sum != args.batch_size:
        raise ValueError(f"Sum(n_detect_pos/n_detect_neg/n_update_*)={mix_sum} != batch_size={args.batch_size}")

    # -------- build datasets --------
    train_cfg = V3DatasetConfig(
        jsonl_path=args.train_jsonl,
        model_name_or_path=args.model_name_or_path,
        num_images=args.num_images,
    )
    val_cfg = V3DatasetConfig(
        jsonl_path=args.val_jsonl,
        model_name_or_path=args.model_name_or_path,
        num_images=args.num_images,
    )

    train_ds = HelmJsonlDatasetV3(train_cfg)
    val_ds = HelmJsonlDatasetV3(val_cfg)

    # label stats (for sanity)
    train_counts = {k: len(v) for k, v in train_ds.get_pools().items()}
    val_counts = {k: len(v) for k, v in val_ds.get_pools().items()}
    print("[TRAIN label counts]", train_counts)
    print("[VAL   label counts]", val_counts)

    # eval subset
    if args.eval_max_samples and args.eval_max_samples > 0 and args.eval_max_samples < len(val_ds):
        rng = random.Random(args.eval_seed)
        idxs = list(range(len(val_ds)))
        rng.shuffle(idxs)
        idxs = idxs[: args.eval_max_samples]
        val_ds = Subset(val_ds, idxs)
        print(f"[VAL] using subset: {len(idxs)} samples")

    # -------- steps_per_epoch --------
    if args.steps_per_epoch and args.steps_per_epoch > 0:
        steps_per_epoch = int(args.steps_per_epoch)
    else:
        steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    print(f"[TRAIN] steps_per_epoch={steps_per_epoch} (len(train)={len(train_ds)}, batch={args.batch_size})")

    # -------- sampler --------
    pools = train_ds.get_pools()
    sampler = MixedBatchSampler(
        pools=pools,
        per_batch=per_batch,
        steps_per_epoch=steps_per_epoch,
        seed=args.sampler_seed,
        with_replacement=bool(args.with_replacement),
        shuffle_within_batch=True,
    )

    # -------- model / collator --------
    model = build_model(
        model_name_or_path=args.model_name_or_path,
        use_qlora=bool(args.use_qlora),
        bf16=bool(args.bf16),
        attn_impl=args.attn_impl,
    )

    collator = DataCollatorForQwenVL(train_ds.processor)

    # -------- HF training args --------
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_train_batch_size=1,  # 무시됨(우리는 batch_sampler 사용)
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=bool(args.bf16),
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = MixedBatchTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[ParamCountCallback()],
        train_batch_sampler=sampler,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()