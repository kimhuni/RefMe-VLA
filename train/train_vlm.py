# train_vlm.py
import os
import math
import argparse
from typing import Optional
import glob
import inspect

import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataset_loader import VLMJSONDataset, DataCollatorVLM

# (선택) QLoRA용 bitsandbytes 설정
try:
    from transformers import BitsAndBytesConfig
    _BNB = True
except Exception:
    _BNB = False

# W&B 콜백
from transformers.integrations import WandbCallback
import wandb


# ===== 1) 유틸: 파라미터 수 계산 =====
def count_parameters(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ===== 2) W&B 커스텀 콜백 (lr 및 파라미터 수 로깅) =====
class LrAndParamCallback(WandbCallback):
    def __init__(self, trainable_params: int, total_params: int):
        super().__init__()
        self.trainable_params = trainable_params
        self.total_params = total_params

    def on_train_begin(self, args, state, control, **kwargs):
        # 파라미터 수 summary 기록
        if wandb.run is not None:
            wandb.summary["trainable_params"] = self.trainable_params
            wandb.summary["total_params"] = self.total_params

    def on_log(self, args, state, control, logs=None, **kwargs):
        # HF Trainer가 logs에 learning_rate, loss를 넣어줌 → W&B에 그대로 기록
        if wandb.run is not None and logs is not None:
            wandb.log(logs, step=state.global_step)


class DebugTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, **kwargs):
        # Print once per process, for the first few steps
        step = getattr(self.state, "global_step", 0)
        dbg_once = os.environ.get("VLM_DEBUG_FORWARD_ONCE") == "1"
        dbg_all  = os.environ.get("VLM_DEBUG_FORWARD_ALL") == "1"
        if (dbg_all or (dbg_once and step == 0)):
            ig = inputs.get("image_grid_thw", inputs.get("grid_thw"))
            dbg_ids = inputs.get("debug_ids")
            if ig is not None:
                try:
                    # show per-sample entry lengths
                    lens = [[len(e) if hasattr(e, "__len__") else -1 for e in s] for s in ig]
                    print(f"[DebugTrainer] step={step} grid_thw lens per-sample={lens[:2]}")
                    # print first offending entry if any
                    for bi, sample in enumerate(ig):
                        for gi, entry in enumerate(sample):
                            ok = isinstance(entry, (list, tuple)) and len(entry) == 3
                            if not ok:
                                sid = (dbg_ids[bi] if isinstance(dbg_ids, list) and bi < len(dbg_ids) else f"idx{bi}")
                                print(f"[DebugTrainer][BAD] sample={sid} entry_idx={gi} value={entry}")
                                raise ValueError("Bad grid_thw passed to model forward")
                except Exception as e:
                    print(f"[DebugTrainer] Exception while inspecting inputs: {e}")
                    raise
        return super().compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)


def expand_jsonl_paths(path: str) -> list[str]:
    if os.path.isfile(path) and path.endswith(".jsonl"):
        return [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)
        return sorted(files)
    else:
        raise ValueError(f"Path {path} is neither a .jsonl file nor a directory containing .jsonl files.")

# ===== 3) 메인 =====
def main():
    parser = argparse.ArgumentParser(description="VLM Fine-tuning (QwenVL / InternVL) with LoRA/QLoRA")
    # 필수
    parser.add_argument("--model_name", type=str, required=True,
                        help="예) Qwen/Qwen2.5-VL-7B-Instruct 또는 OpenGVLab/InternVL3-2B")
    parser.add_argument("--train_path", type=str, required=True,
                        help="학습용 jsonl 파일 또는 jsonl shards가 있는 디렉토리 (예: .../train_final)")
    parser.add_argument("--val_path", type=str, required=True,
                        help="검증용 jsonl 파일 또는 jsonl shards가 있는 디렉토리 (예: .../eval_final)")
    parser.add_argument("--output_dir", type=str, required=True)

    # 학습 하이퍼파라미터
    # parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    # 분산/가속 옵션 (필요시 그대로 전달)
    parser.add_argument("--deepspeed", type=str, default=None, help="deepspeed json 경로")
    parser.add_argument("--fsdp", type=str, default=None, help='예: "full_shard auto_wrap"')
    parser.add_argument("--fsdp_config", type=str, default=None)

    # LoRA/QLoRA
    # Adapter selector
    parser.add_argument("--adapter", type=str, default="lora", choices=["none", "lora", "qlora"],
                        help="어댑터 종류 선택: none, lora, qlora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj",
                        help="콤마 구분, 모듈명은 모델에 따라 다를 수 있음")
    parser.add_argument("--include_mm_projector", action="store_true", default=False,
                        help="가능한 경우 멀티모달 projector에도 LoRA 적용")

    # 이미지/토크나이저 옵션
    parser.add_argument("--image_key", type=str, default="image_path")

    # W&B
    parser.add_argument("--wandb_project", type=str, default="vlm-finetune")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_enable", type=bool, default=True)

    args = parser.parse_args()

    # ===== W&B 초기화 =====
    if args.wandb_enable:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name is None:
            args.wandb_run_name = f"{os.path.basename(args.output_dir) if hasattr(args, 'output_dir') else 'run'}"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # ===== Processor/Model 로드 =====
    processor = AutoProcessor.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    quant_config = None
    if args.adapter == "qlora":
        if not _BNB:
            raise RuntimeError("QLoRA를 위해서는 bitsandbytes가 필요합니다. (pip install bitsandbytes)")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        device_map = None
    else:
        device_map = "auto" if (args.adapter == "qlora" or torch.cuda.device_count() == 1) else None

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
        quantization_config=quant_config,
        attn_implementation="flash_attention_2"
    )

    # gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("[Setup] Configuring tokenizer special tokens...")
    tokenizer = processor.tokenizer

    new_special_tokens = []

    # 1. <image> 토큰 추가 시도
    if "<image>" not in tokenizer.all_special_tokens:
        print("Adding '<image>' to special tokens.")
        new_special_tokens.append("<image>")
    else:
        print("'<image>' token already exists.")

    # 2. Pad 토큰 확인 및 설정
    pad_token_id = tokenizer.pad_token_id
    pad_token = tokenizer.pad_token

    if pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print("pad_token_id is None. Setting pad_token = eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            pad_token_id = tokenizer.eos_token_id
        else:
            # EOS 토큰도 없는 비상 상황
            print("pad_token_id and eos_token_id are None. Adding '[PAD]' as pad_token.")
            pad_token = "[PAD]"  # 사용할 토큰 문자열
            if pad_token not in tokenizer.all_special_tokens:
                new_special_tokens.append(pad_token)
            tokenizer.pad_token = pad_token
            # pad_token_id는 add_special_tokens 이후에 확정

    # 3. 새로운 토큰이 있으면 *먼저* 토크나이저에 추가
    if new_special_tokens:
        print(f"Adding special tokens: {new_special_tokens}")
        # add_special_tokens는 새 토큰만 추가하도록 함
        unique_new_tokens = list(set(new_special_tokens) - set(tokenizer.all_special_tokens))
        if unique_new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": unique_new_tokens})

    # 4. [중요] 토큰 추가 후 pad_token_id 최종 확정
    if pad_token_id is None:
        pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    # 5. [핵심 수정] 모델 임베딩 리사이즈를 *항상* 실행
    print(f"Resizing model token embeddings to length: {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

    # 6. [핵심 수정] 모델 config에도 pad_token_id를 명시적으로 설정
    if model.config.pad_token_id != pad_token_id:
        print(f"Updating model.config.pad_token_id to: {pad_token_id}")
        model.config.pad_token_id = pad_token_id

    print(f"Tokenizer configured. pad_token_id={pad_token_id}")

    # ===== LoRA / QLoRA 적용 =====
    if args.adapter == "lora":
        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        if args.include_mm_projector:
            # projector 모듈 이름은 모델마다 다를 수 있음: 필요시 여기에 추가
            # 예) target_modules += ["mm_projector", "vision_projection"]
            target_modules = list(set(target_modules + ["mm_projector", "vision_projection"]))

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

    # ===== 데이터셋/콜레이터 =====
    train_files = expand_jsonl_paths(args.train_path)
    val_files = expand_jsonl_paths(args.val_path)

    # [핵심 수정] processor (객체) 대신 args.model_name (문자열)을 전달
    train_ds = VLMJSONDataset(train_files, args.model_name, image_key=args.image_key)
    val_ds = VLMJSONDataset(val_files, args.model_name, image_key=args.image_key)

    # [수정] collator는 V4와 동일 (processor가 필요 없음)
    collator = DataCollatorVLM(
        pad_token_id=pad_token_id
    )

    # ===== 학습 파라미터/로깅 설정 =====
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)

    print("---" * 20)
    print(f"Total Parameters:     {total_params / 1_000_000:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1_000_000:.2f} M")
    print(f"Trainable Ratio:      {trainable_params / total_params * 100:.4f} %")
    print("---" * 20)

    # Build TrainingArguments kwargs dynamically for compatibility
    ta_init_params = inspect.signature(TrainingArguments.__init__).parameters
    ta_kwargs = dict(
        output_dir=args.output_dir if hasattr(args, 'output_dir') else "./output",
        num_train_epochs=1000,  # 큰 수로 설정하여 HF가 epoch 계산하도록 함
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=6,
        bf16=args.bf16,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        seed=args.seed,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp,
        fsdp_config=args.fsdp_config,
        gradient_checkpointing=args.gradient_checkpointing,
        max_steps=args.train_steps,
    )

    # Handle evaluation strategy param name compatibility
    if "evaluation_strategy" in ta_init_params:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in ta_init_params:
        ta_kwargs["eval_strategy"] = "steps"

    # Handle save strategy param name compatibility
    if "save_strategy" in ta_init_params:
        ta_kwargs["save_strategy"] = "steps"

    if "dataloader_num_workers" in ta_init_params:
        # A6000 서버의 CPU 코어 수에 맞춰 적절히 조절 (예: 8, 16, 32)
        # 시작은 8이나 16 정도로 해보세요.
        ta_kwargs["dataloader_num_workers"] = 16

    # Add remove_unused_columns=False to keep debug fields in data collator if supported
    if "remove_unused_columns" in ta_init_params:
        ta_kwargs["remove_unused_columns"] = False


    # Remove keys that are not accepted by current TrainingArguments version
    valid_keys = set(ta_init_params.keys())
    ta_kwargs = {k: v for k, v in ta_kwargs.items() if k in valid_keys}

    args_hf = TrainingArguments(**ta_kwargs)

    # ===== Trainer 구성/학습 =====
    # trainer = Trainer(
    tr_init_params = inspect.signature(Trainer.__init__).parameters
    use_processing_class = "processing_class" in tr_init_params

    trainer_kwargs = dict(
        model=model,
        args=args_hf,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        # tokenizer=processor.tokenizer,
        callbacks=[LrAndParamCallback(trainable_params, total_params)],
    )

    if use_processing_class:
        trainer_kwargs["processing_class"] = processor
    else:
        trainer_kwargs["tokenizer"] = processor.tokenizer

    use_debug_trainer = os.environ.get("VLM_USE_DEBUG_TRAINER") == "1"
    if use_debug_trainer:
        trainer = DebugTrainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)


    trainer.train()

    # 마무리
    if args.wandb_enable:
        wandb.finish()


if __name__ == "__main__":
    main()