# ⚙️ utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def add_special_tokens(tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> None:
    """
    Adds <image> token and pad token to the tokenizer and resizes model embeddings.
    """
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_IMAGE_TOKEN = "<image>"

    special_tokens_dict = {}

    # 1. Add Pad token
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    # 2. Add <image> token (required for Qwen-VL)
    if DEFAULT_IMAGE_TOKEN not in tokenizer.get_vocab():
        special_tokens_dict["additional_special_tokens"] = [DEFAULT_IMAGE_TOKEN]

    if special_tokens_dict:
        print(f"Adding special tokens: {special_tokens_dict}")
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        # Explicitly set pad_token_id in model config
        if "pad_token" in special_tokens_dict:
            model.config.pad_token_id = tokenizer.pad_token_id
            if hasattr(model, "generation_config"):
                model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 3. Synchronize model's image_token_index with tokenizer
    if hasattr(tokenizer, "image_token_index") and hasattr(model.config, "image_token_index"):
        model.config.image_token_index = tokenizer.image_token_index

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model embedding size: {model.get_input_embeddings().weight.shape[0]}")


def get_and_print_trainable_parameters(model) -> (int, int, float):
    """
    Calculates, prints, and returns the number of trainable parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    percentage = 100 * trainable_params / all_param
    print(
        f"✅ Trainable parameters: {trainable_params} || All parameters: {all_param} || "
        f"Trainable %: {percentage:.2f}"
    )

    return trainable_params, all_param, percentage