"""
export PYTHONPATH=$(pwd)
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=7 python train/train_helm_video/train_helm_video_update_dataloader.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/helm_video_task_10/merged/all_train.jsonl \
  --val_jsonl /data/ghkim/helm_data/helm_video_task_10/merged/all_val.jsonl \
  --output_dir /backups/ghkim/HLP_HeLM_video/HeLM_video_task_10_0125 \
  --batch_size 3 \
  --n_detect_pos 1 --n_detect_neg 1 --n_update 1 \
  --max_pixels 602112 \
  --max_steps 30000 \
  --save_steps 500 \
  --logging_steps 10 \
  --wandb_project "RefMe" \
  --wandb_run_name "HeLM_video_task_10_0125"

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 train/train_helm_video/train_helm_video_update_dataloader.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --train_jsonl /data/ghkim/helm_data/helm_video_task_10/merged/all_train.jsonl \
  --val_jsonl /data/ghkim/helm_data/helm_video_task_10/merged/all_val.jsonl \
  --output_dir /backups/ghkim/HLP_HeLM_video/HeLM_video_task_10_0125 \
  --detect_batch_size 8 \
  --update_batch_size 2 \
  --max_pixels 100352 \
  --max_steps 10000 \
  --save_steps 500 \
  --logging_steps 10 \
  --wandb_project "RefMe" \
  --wandb_run_name "HeLM_video_task_10_0125"
"""
# train/train_video/train_video.py
from __future__ import annotations

import argparse
import os
import random
import math
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, BatchSampler
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 위에서 작성한 dataset 모듈 import
from train.train_helm_video.helm_video_dataset import (
    VideoJsonlDataset,
    VideoDatasetConfig,
    DataCollatorForVideoBaseline,
)


# -----------------------------------------------------------------------------
# AlternatingBatchSampler: Detect 배치와 Update 배치를 번갈아 생성
# -----------------------------------------------------------------------------
class AlternatingBatchSampler(BatchSampler):
    def __init__(self, dataset, detect_batch_size=8, update_batch_size=1, seed=42):
        # 1. 데이터셋에서 인덱스 풀 가져오기
        self.pools = dataset.get_pools()

        # 'detect' 통합 (pos + neg)
        self.detect_indices = self.pools.get('detect_pos', []) + self.pools.get('detect_neg', [])
        self.update_indices = self.pools.get('update', [])

        self.detect_batch_size = detect_batch_size
        self.update_batch_size = update_batch_size
        self.seed = seed
        self.epoch = 0

        # 배치 개수 계산
        self.n_detect_batches = (
                                            len(self.detect_indices) + detect_batch_size - 1) // detect_batch_size if detect_batch_size > 0 else 0
        self.n_update_batches = (
                                            len(self.update_indices) + update_batch_size - 1) // update_batch_size if update_batch_size > 0 else 0
        self.num_batches = self.n_detect_batches + self.n_update_batches

        if self.num_batches == 0:
            raise ValueError("No batches created. Check if dataset is empty or batch sizes are 0.")

    def __iter__(self):
        # 매 에폭마다 셔플링을 위해 시드 조정
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 인덱스 셔플 (Python random 대신 torch generator 사용 권장, 여기서는 간단히 리스트 셔플)
        # 멀티 프로세스/DDP 환경에서는 set_epoch를 통해 시드를 동기화해야 함.
        # 여기서는 간단한 구현을 위해 random 사용 (단일 GPU/노드 기준)
        random.seed(self.seed + self.epoch)

        d_indices = self.detect_indices[:]
        u_indices = self.update_indices[:]
        random.shuffle(d_indices)
        random.shuffle(u_indices)

        # 배치 생성
        detect_batches = [
            d_indices[i: i + self.detect_batch_size]
            for i in range(0, len(d_indices), self.detect_batch_size)
        ]

        update_batches = [
            u_indices[i: i + self.update_batch_size]
            for i in range(0, len(u_indices), self.update_batch_size)
        ]

        # 인터리빙 (섞기)
        all_batches = detect_batches + update_batches
        random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

        self.epoch += 1

    def __len__(self):
        return self.num_batches


# -----------------------------------------------------------------------------
# Custom Trainer: Sampler 주입
# -----------------------------------------------------------------------------
class CustomTrainer(Trainer):
    def __init__(self, *args, detect_batch_size=8, update_batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.detect_batch_size = detect_batch_size
        self.update_batch_size = update_batch_size

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = AlternatingBatchSampler(
            self.train_dataset,
            detect_batch_size=self.detect_batch_size,
            update_batch_size=self.update_batch_size
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    # 모델 및 데이터 경로
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # [수정] 배치 설정: Detect와 Update 각각의 배치 사이즈
    parser.add_argument("--detect_batch_size", type=int, default=8, help="Batch size for DETECT samples")
    parser.add_argument("--update_batch_size", type=int, default=1,
                        help="Batch size for UPDATE samples (keep small for VRAM)")

    # 학습 스텝 및 에폭
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)

    # 학습 하이퍼파라미터
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=602112)  # 224*224 ~ 640*640 수준 권장
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # WandB
    parser.add_argument("--wandb_project", type=str, default="HeLM_Video_Baseline")
    parser.add_argument("--wandb_run_name", type=str, default="")

    args = parser.parse_args()

    # 1. WandB 환경 설정
    os.environ["WANDB_PROJECT"] = args.wandb_project
    if args.wandb_run_name:
        os.environ["WANDB_NAME"] = args.wandb_run_name

    # 2. 데이터셋 로드
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

    import random
    if len(val_ds.rows) > 200:
        print(f"[Info] Subsampling val dataset: {len(val_ds.rows)} -> 200")
        # 랜덤하게 200개만 선택
        val_ds.rows = random.sample(val_ds.rows, 200)
        # 주의: val_ds는 Sampler를 안 쓰므로 pools 갱신 안 해도 무방함

    print(f"[Data Info] Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")

    print(f"[Data Info] Train Size: {len(train_ds)}, Val Size: {len(val_ds)}")

    # 3. 모델 준비 (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Data Collator 정의 (이 부분이 누락되어 있었음)
    data_collator = DataCollatorForVideoBaseline(train_ds.processor)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,

        # 여기서는 Trainer의 DataLoader가 주는 배치를 그대로 씀 (Sampler가 이미 배치 단위로 줌)
        # per_device_train_batch_size는 로깅용으로 사용되거나 무시됨
        per_device_train_batch_size=args.detect_batch_size,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        report_to="wandb",
        run_name=args.wandb_run_name,

        bf16=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    # 6. Trainer 초기화
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,  # [수정] 정의된 collator 전달
        # Custom Args 전달
        detect_batch_size=args.detect_batch_size,
        update_batch_size=args.update_batch_size
    )

    print("🚀 Starting Training...")
    trainer.train()

    print("💾 Saving Model...")
    trainer.save_model(args.output_dir)
    print("✅ Done!")


if __name__ == "__main__":
    main()