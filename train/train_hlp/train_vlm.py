# 🚅 train_vlm.py
"""
echo "--- 🚀 Starting Full Finetuning (logging to W&B) ---"
export WANDB_NAME="qwen-vlm-full-run-1"
python train_vlm.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
    --dataset_dir ${DATASET_ROOT_DIR} \
    --output_dir "./results/qwen_vlm_full_finetuned" \
    --max_steps 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --dataloader_num_workers 16 \
    --bf16 True \
    --gradient_checkpointing True \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 100 \
    --report_to "wandb"
"""
# 🚅 train_vlm.py
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List
import logging

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

# Local imports
from dataset_vlm import VlmDataset, DataCollatorForVLM
from utils import add_special_tokens, get_and_print_trainable_parameters

# Logger
logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingArgs:
    """
    Custom arguments for the training script.
    """
    # --- Model and Data Paths ---
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        metadata={"help": "Hugging Face path to the VLM to finetune."}
    )
    dataset_dir: str = field(
        metadata={"help": "Root directory of the dataset containing sharded .jsonl files (e.g., /data/my_dataset)"}
    )
    output_dir: str = field(
        default="./results",
        metadata={"help": "Directory where model checkpoints and results will be saved."}
    )

    # --- Training Strategy ---
    bf16: bool = field(
        default=True,
        metadata={"help": "Whether to use bf16 training (recommended for Ampere GPUs)."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Whether to use gradient checkpointing to save memory."}
    )
    dataloader_num_workers: int = field(
        default=16,
        metadata={"help": "[v5 Architecture] Number of parallel workers for data loading."}
    )

    # --- PEFT (LoRA/QLoRA) Config ---
    # [Modified] Default is False (Full Finetuning)
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use standard LoRA (default is Full Finetuning)."}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA (4-bit quantized LoRA)."}
    )

    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank (r)."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha (alpha)."}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."}
    )
    lora_target_modules: str = field(
        default="c_attn,attn.c_proj,w1,w2",
        metadata={"help": "Modules to apply LoRA to (comma-separated)."}
    )


def main():
    # 1. Parse arguments (TrainingArguments + ModelTrainingArgs)
    parser = HfArgumentParser((TrainingArguments, ModelTrainingArgs))
    training_args, model_args = parser.parse_args_into_dataclasses()

    # [Modified] Argument validation
    if model_args.use_lora and model_args.use_qlora:
        raise ValueError("Error: --use_lora and --use_qlora cannot be used simultaneously. "
                         "For QLoRA, only specify --use_qlora.")

    # 2. Configure QLoRA (BitsAndBytes) [Enabled only if use_qlora is True]
    bnb_config = None
    if model_args.use_qlora:
        print("Activating QLoRA (4-bit quantization).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # 3. Load Model
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,  # bnb_config is applied only if use_qlora=True
        torch_dtype=torch.bfloat16 if model_args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )

    # 4. Load Processor (Tokenizer)
    print(f"Loading processor: {model_args.model_name_or_path}")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer = processor.tokenizer

    # 5. Add Special Tokens and Resize Embeddings
    add_special_tokens(tokenizer, model)

    # 6. [Modified] Conditionally setup PEFT (LoRA/QLoRA)
    if model_args.use_lora or model_args.use_qlora:
        peft_method = "QLoRA" if model_args.use_qlora else "LoRA"
        print(f"✅ Setting up PEFT ({peft_method})...")

        peft_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules.split(','),
            task_type="CAUSAL_LM",
            bias="none"
        )

        # Enable gradient checkpointing on the base model only for standard LoRA
        if model_args.gradient_checkpointing and not model_args.use_qlora:
            model.gradient_checkpointing_enable()

        model = get_peft_model(model, peft_config)

    else:
        print("✅ Running Full Finetuning (PEFT not enabled).")
        # Enable gradient checkpointing for full finetuning
        if model_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    # 7. Print trainable parameters (and get values)
    trainable_params, all_params, percentage = get_and_print_trainable_parameters(model)

    # 8. Load Dataset (v5 Architecture)
    print(f"Loading dataset from directory: {model_args.dataset_dir}")
    train_dataset = VlmDataset(
        dataset_dir=model_args.dataset_dir,
        model_name_or_path=model_args.model_name_or_path
    )

    # 9. Initialize Data Collator
    data_collator = DataCollatorForVLM(tokenizer=tokenizer)

    # 10. Configure Trainer Settings
    training_args.gradient_checkpointing = model_args.gradient_checkpointing
    training_args.bf16 = model_args.bf16
    training_args.output_dir = model_args.output_dir
    training_args.dataloader_num_workers = model_args.dataloader_num_workers
    training_args.remove_unused_columns = False
    # training_args.report_to is set via the command line

    # 11. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 12. [Added] Log custom parameters to W&B (if enabled)
    # This logs the values once at the beginning of the run.
    if training_args.report_to and "wandb" in training_args.report_to:
        trainer.log({
            "trainable_params": trainable_params,
            "all_params": all_params,
            "trainable_percentage": percentage
        })

    # 13. Start Training
    print("🚀 Starting training...")
    trainer.train()

    # 14. [Modified] Conditionally save the final model/adapter
    print("Training finished. Saving final model/adapter...")

    if model_args.use_lora or model_args.use_qlora:
        # For LoRA or QLoRA: Save only the adapter
        final_adapter_path = os.path.join(training_args.output_dir, "final-adapter")
        model.save_pretrained(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        print(f"✅ Adapter and tokenizer saved to: {final_adapter_path}")
    else:
        # For Full Finetuning: Save the entire model
        final_model_path = os.path.join(training_args.output_dir, "final-full-model")
        trainer.save_model(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"✅ Full model and tokenizer saved to: {final_model_path}")


if __name__ == "__main__":
    main()