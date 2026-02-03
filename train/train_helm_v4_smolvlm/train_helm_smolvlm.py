# train_helm_smolvlm.py
from __future__ import annotations
import argparse
import os
import random
from typing import Dict, List, Optional, Any
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from collections import Counter
import json

# helm_dataset.py 파일에 정의된 클래스들을 가져옵니다.
from helm_dataset import (
    HelmJsonlDatasetSmolVLM,
    SmolVLMDatasetConfig,
    DataCollatorForSmolVLM,
)

"""
# run_train.sh
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=6 python train/train_helm_v4_smolvlm/train_helm_smolvlm.py \
    --model_name_or_path "/ckpt/SmolVLM-500M-Instruct" \
    --train_jsonl "/data/ghkim/helm_data/helm_v4_task_10/merged/all_train.jsonl" \
    --val_jsonl "/data/ghkim/helm_data/helm_v4_task_10/merged/all_val.jsonl" \
    --num_images 1 \
    --output_dir "/backups/ghkim/HeLM_v4/HLP_HeLM_v4_SmolVLM_Full_FT_v4_task_10" \
    --batch_size 24 \
    --n_detect_pos 8 \
    --n_detect_neg 8 \
    --n_update_intra 6 \
    --n_update_transition 2 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --bf16 True \
    --attn_impl "flash_attention_2" \
    --eval_max_samples 40 \
    --wandb_project "RefMe" \
    --wandb_run_name "HLP_HeLM_v4_SmolVLM_Full_FT_v4_task_10"
"""


def print_label_stats(jsonl_path):
    counter = Counter()
    with open(jsonl_path, "r") as f:
        for line in f:
            row = json.loads(line)
            counter[row["label"]] += 1

    total = sum(counter.values())
    print("\n[Train data label distribution]")
    for k, v in counter.items():
        print(f"  {k:>18}: {v:6d} ({v / total:.2%})")
    print(f"  {'TOTAL':>18}: {total:6d}\n")


# ---------------------------
# Mixed Batch Sampler
# ---------------------------
class MixedBatchSampler:
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

        for k, n in self.per_batch.items():
            if n < 0: raise ValueError(f"per_batch[{k}] must be >=0")
            if n > 0 and len(self.pools.get(k, [])) == 0:
                raise ValueError(f"pool '{k}' is empty but per_batch[{k}]={n}")

    def __len__(self):
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return (self.steps_per_epoch + world - 1) // world

    def __iter__(self):
        rng = random.Random(self.seed)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        if self.with_replacement:
            for step in range(self.steps_per_epoch):
                batch: List[int] = []
                for k, n in self.per_batch.items():
                    if n > 0: batch.extend(rng.choices(self.pools[k], k=n))
                if self.shuffle_within_batch: rng.shuffle(batch)
                if (step % world) == rank: yield batch
        else:
            working = {k: list(v) for k, v in self.pools.items()}
            for k in working: rng.shuffle(working[k])
            for step in range(self.steps_per_epoch):
                batch: List[int] = []
                for k, n in self.per_batch.items():
                    if n == 0: continue
                    if len(working[k]) < n: return
                    batch.extend(working[k][:n])
                    del working[k][:n]
                if self.shuffle_within_batch: rng.shuffle(batch)
                if (step % world) == rank: yield batch


class MixedBatchTrainer(Trainer):
    def __init__(self, *args, train_batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._train_batch_sampler = train_batch_sampler

    def get_train_dataloader(self):
        if self._train_batch_sampler is None: return super().get_train_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._train_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )


# ---------------------------
# Training Utils
# ---------------------------
def count_parameters(model) -> tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100.0 * trainable / max(total, 1)


class ParamCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        trainer, model = kwargs.get("trainer"), kwargs.get("model")
        if trainer and model:
            trainable, total, pct = count_parameters(model)
            trainer.log({"trainable_params": trainable, "all_params": total, "trainable_pct": pct})


def build_model(model_name_or_path: str, bf16: bool, attn_impl: str):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available(): torch.cuda.set_device(local_rank)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        _attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    for param in model.parameters():
        param.requires_grad = True  # Full Fine-tuning

    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, required=True)
    ap.add_argument("--val_jsonl", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_images", type=int, default=1)

    # Batch mixture knobs
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_detect_pos", type=int, default=2)
    ap.add_argument("--n_detect_neg", type=int, default=2)
    ap.add_argument("--n_update_intra", type=int, default=4)
    ap.add_argument("--n_update_transition", type=int, default=0)
    ap.add_argument("--with_replacement", type=lambda x: (str(x).lower() == 'true'), default=True)
    ap.add_argument("--sampler_seed", type=int, default=0)

    # Epoch sizing
    ap.add_argument("--steps_per_epoch", type=int, default=0)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)

    # Eval sampling
    ap.add_argument("--eval_max_samples", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=123)

    # Training args
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=300)
    ap.add_argument("--eval_steps", type=int, default=50)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--dataloader_num_workers", type=int, default=4)
    ap.add_argument("--bf16", type=lambda x: (str(x).lower() == 'true'), default=True)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2")

    # WandB
    ap.add_argument("--wandb_project", type=str, default="RefMe")
    ap.add_argument("--wandb_run_name", type=str, default="")

    args = ap.parse_args()

    # WandB setup
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name: os.environ["WANDB_NAME"] = args.wandb_run_name

    # Validate batch
    per_batch = {"detect_pos": args.n_detect_pos, "detect_neg": args.n_detect_neg,
                 "update_intra": args.n_update_intra, "update_transition": args.n_update_transition}
    if sum(per_batch.values()) != args.batch_size:
        raise ValueError(f"Batch sum {sum(per_batch.values())} != batch_size {args.batch_size}")

    # Build datasets
    train_cfg = SmolVLMDatasetConfig(args.train_jsonl, args.model_name_or_path, args.num_images)
    val_cfg = SmolVLMDatasetConfig(args.val_jsonl, args.model_name_or_path, args.num_images)
    train_ds, val_ds = HelmJsonlDatasetSmolVLM(train_cfg), HelmJsonlDatasetSmolVLM(val_cfg)

    # Eval subsetting
    if 0 < args.eval_max_samples < len(val_ds):
        rng = random.Random(args.eval_seed)
        val_ds = Subset(val_ds, rng.sample(range(len(val_ds)), args.eval_max_samples))

    # Sampler setup
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else max(1, len(train_ds) // args.batch_size)
    sampler = MixedBatchSampler(train_ds.get_pools(), per_batch, steps_per_epoch,
                                args.sampler_seed, args.with_replacement)

    # Model & Trainer
    model = build_model(args.model_name_or_path, args.bf16, args.attn_impl)
    collator = DataCollatorForSmolVLM(train_ds.processor)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        bf16=args.bf16,
        gradient_checkpointing=True,
        report_to=["wandb"],
        remove_unused_columns=False,
    )

    trainer = MixedBatchTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, callbacks=[ParamCountCallback()], train_batch_sampler=sampler
    )

    print_label_stats(args.train_jsonl)
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()