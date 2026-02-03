# train/train_video/train_video.py
from __future__ import annotations

# 위에서 작성한 dataset 모듈 import
from train.train_helm_video.helm_video_dataset import (
    VideoJsonlDataset,
    VideoDatasetConfig,
    DataCollatorForVideoBaseline,
)

"""
export PYTHONPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=7 python train/train_helm_video/train_helm_video.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/helm_video_task_10/merged/all_train.jsonl \
  --val_jsonl /data/ghkim/helm_data/helm_video_task_5/merged/all_val.jsonl \
  --output_dir /backups/ghkim/HLP_HeLM_video/HeLM_video_task_5_0125 \
  --batch_size 4 \
  --n_detect_pos 2 --n_detect_neg 1 --n_update 1 \
  --max_pixels 602112 \
  --max_steps 30000 \
  --save_steps 500 \
  --logging_steps 10 \
  --wandb_project "RefMe" \
  --wandb_run_name "HeLM_video_task_5_0125"


CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 train/train_helm_video/train_helm_video.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/press_button_in_order/extended/visual_memory_jsonl/train.jsonl \
  --val_jsonl /data/ghkim/helm_data/press_button_in_order/extended/visual_memory_jsonl/val.jsonl \
  --output_dir /backups/ghkim/HLP_HeLM_video/HeLM_video_press_button_in_order_extended_0131 \
  --batch_size 1 \
  --n_detect_pos 0 --n_detect_neg 0 --n_update 1 \
  --max_pixels 310000 \
  --max_steps 30000 \
  --save_steps 500 \
  --gradient_accumulation_steps 4 \
  --logging_steps 10 \
  --wandb_project "RefMe" \
  --wandb_run_name "HeLM_video_press_button_in_order_extended_0131"
"""

import argparse
import os
import random
import math
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------------------------------------------------------------
# MixedBatchSampler: Detect와 Update 데이터를 비율대로 섞어서 배치를 만듦
# -----------------------------------------------------------------------------
class MixedBatchSampler:
    def __init__(self, pools: Dict[str, List[int]], per_batch: Dict[str, int], steps_per_epoch: int, seed: int = 0):
        self.pools = {k: list(v) for k, v in pools.items()}
        self.per_batch = per_batch
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed

    def __len__(self):
        # DDP 환경 고려: 전체 스텝 수 / GPU 개수
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        return (self.steps_per_epoch + world_size - 1) // world_size

    def __iter__(self):
        rng = random.Random(self.seed)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        # 전체 Epoch 동안 필요한 배치 생성
        for step in range(self.steps_per_epoch):
            batch_indices = []

            # 각 라벨별로 할당된 개수만큼 랜덤 추출
            for label, count in self.per_batch.items():
                if count > 0 and label in self.pools and self.pools[label]:
                    batch_indices.extend(rng.choices(self.pools[label], k=count))

            # 배치 내부 셔플 (Detect와 Update가 섞이도록)
            rng.shuffle(batch_indices)

            # DDP: 현재 GPU(Rank)가 처리할 배치만 yield
            if (step % world_size) == rank:
                yield batch_indices


# -----------------------------------------------------------------------------
# Custom Trainer: Sampler를 적용하기 위해 get_train_dataloader 오버라이딩
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # 모델 및 데이터 경로
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to merged train.jsonl")
    parser.add_argument("--val_jsonl", type=str, required=True, help="Path to merged val.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output checkpoint directory")

    # 배치 구성 (Batch Composition)
    # 총합이 실제 batch_size가 됨
    parser.add_argument("--batch_size", type=int, default=8, help="Total batch size per step")
    parser.add_argument("--n_detect_pos", type=int, default=2)
    parser.add_argument("--n_detect_neg", type=int, default=2)
    parser.add_argument("--n_update", type=int, default=4, help="Number of update samples per batch")

    # 학습 스텝 및 에폭 (Step Control)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform. Overrides num_train_epochs.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log to wandb every X steps")

    # 학습 하이퍼파라미터
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=12845056, help="Max pixels limit for VRAM management")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # WandB 설정
    parser.add_argument("--wandb_project", type=str, default="HeLM_Video_Baseline", help="WandB Project Name")
    parser.add_argument("--wandb_run_name", type=str, default="", help="WandB Run Name (optional)")

    args = parser.parse_args()

    # 1. WandB 환경 설정
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # 2. 데이터셋 로드
    # Dataset Config 생성
    train_cfg = VideoDatasetConfig(
        jsonl_path=args.train_jsonl,
        model_name_or_path=args.model_name_or_path,
        max_pixels=args.max_pixels
    )
    val_cfg = VideoDatasetConfig(
        jsonl_path=args.val_jsonl,
        model_name_or_path=args.model_name_or_path,
        max_pixels=args.max_pixels
    )

    train_ds = VideoJsonlDataset(train_cfg)
    val_ds = VideoJsonlDataset(val_cfg)

    print(f"[Data Info] Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")

    # 3. 배치 샘플러 설정 (MixedBatchSampler)
    # 배치 구성 검증: 구성요소의 합이 batch_size와 일치해야 함
    per_batch = {
        "detect_pos": args.n_detect_pos,
        "detect_neg": args.n_detect_neg,
        "update": args.n_update
    }
    batch_sum = sum(per_batch.values())
    if batch_sum != args.batch_size:
        raise ValueError(f"Batch composition sum ({batch_sum}) does not match --batch_size ({args.batch_size})")

    # steps_per_epoch 계산 (max_steps 설정 시 고려)
    if args.max_steps > 0:
        total_steps = args.max_steps
        # max_steps가 주어지면 에폭은 무시되지만, 샘플러 동작을 위해 대략적으로 계산
        steps_per_epoch = total_steps
        args.num_train_epochs = 1.0  # Trainer 내부 로직 만족용
    else:
        steps_per_epoch = len(train_ds) // args.batch_size

    print(f"[Sampler] Steps per epoch: {steps_per_epoch}, Batch Mix: {per_batch}")

    sampler = MixedBatchSampler(
        pools=train_ds.get_pools(),
        per_batch=per_batch,
        steps_per_epoch=steps_per_epoch,
    )

    # 4. 모델 준비 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Training Arguments 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,  # max_steps 추가
        learning_rate=args.learning_rate,

        # Batch Size 설정 (중요)
        # Loader가 이미 Batch_size(=8)만큼 묶어서 줌.
        # 따라서 per_device_train_batch_size=1로 설정하여 Trainer가 추가로 묶지 않도록 함.
        per_device_train_batch_size=1,# [훈련] 배치 사이즈 1 (Sampler가 묶어줌)
        # 기본값이 8이라서 터지는 것입니다. 1로 줄여주세요.
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_8bit",

        gradient_checkpointing=True,  # 체크포인팅 활성화
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging & Saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",  # steps 단위 저장
        evaluation_strategy="steps",
        eval_steps=args.save_steps,  # 저장할 때 평가도 같이 수행
        report_to="wandb",  # WandB 활성화
        run_name=args.wandb_run_name,

        # System
        bf16=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,  # Custom Collator 사용 시 필수
        ddp_find_unused_parameters=False,  # DDP 경고 방지
    )

    # 6. Trainer 초기화 및 학습 시작
    trainer = MixedBatchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForVideoBaseline(train_ds.processor),
        train_batch_sampler=sampler
    )

    print("🚀 Starting Training...")
    trainer.train()

    print("💾 Saving Model...")
    trainer.save_model(args.output_dir)
    print("✅ Done!")


if __name__ == "__main__":
    main()