# evaluate/eval_real_time_HLP_qwen.py
import re
import time
from typing import Dict, Any

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


HLP_HEADER = (
    "Role: High-Level Planner (HLP).\n"
    "Given the two images and Previous_Memory, update the memory and choose the next atomic command.\n"
    "- Only advance Progress when the event has occurred in the current frame.\n"
    "- World_State should be concise and persistent (use None if no state).\n"
    "- Command should be either the task command or \"done\" if finished.\n"
    # "- Command should be either the \"press the blue button\" or \"done\" if finished.\n"
    "Return YAML with keys Progress, World_State, Command.\n"
)

_YAML_KEYS = ("Progress", "World_State", "Command")


def parse_hlp_yaml(text: str) -> Dict[str, str]:
    # 모델 출력에 잡다한 텍스트가 섞여도 key 라인만 최대한 추출
    lines = []
    for ln in text.splitlines():
        if any(ln.strip().startswith(k + ":") for k in _YAML_KEYS):
            lines.append(ln)

    raw = "\n".join(lines).strip() if lines else text.strip()

    out = {"Progress": "", "World_State": "None", "Command": ""}

    try:
        d = yaml.safe_load(raw)
        if isinstance(d, dict):
            for k in _YAML_KEYS:
                v = d.get(k, None)
                if v is None:
                    out[k] = "None" if k == "World_State" else ""
                else:
                    out[k] = str(v)
        else:
            raise ValueError("YAML not a dict")
    except Exception:
        # fallback: line-based
        for k in _YAML_KEYS:
            m = re.search(rf"^{k}\s*:\s*(.*)$", raw, flags=re.M)
            if m:
                v = m.group(1).strip()
                out[k] = "None" if v.lower() in ("null", "none", "") else v

    if out["World_State"].lower() == "null":
        out["World_State"] = "None"
    return out


class HLPQwen:
    """
    Qwen2.5-VL + (Q)LoRA adapter inference
    - main에서 processor로 만든 batch를 받아 forward(batch) 수행
    - output은 YAML 파싱 -> dict(Progress/World_State/Command)
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        load_in_4bit: bool = True,
        max_new_tokens: int = 128,
    ):
        self.device = device
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        t0 = time.time()
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        print("[HLP] loaded Qwen 2.5 VL")
        model = PeftModel.from_pretrained(base, adapter_path)
        # inference에서는 merge 권장(속도/단순)
        self.model = model.merge_and_unload().eval()
        print("[HLP] QLoRA merged")

        print(f"[HLP] load done: {time.time()-t0:.2f}s")

    def reset(self):
        # 현재는 HLP 내부 state는 main이 prev_memory로 관리
        pass

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, str]:
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            # do_sample=True,
            # temperature=1e-6,  # 🔥 여기
            # top_p=0.9,
        )

        # prompt 잘라내기
        in_len = inputs["input_ids"].shape[1]
        gen_trim = gen_ids[:, in_len:]

        out_text = self.tokenizer.batch_decode(
            gen_trim,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print("------------ [HLP] raw_text -----------\n", out_text, "-------------------------------")

        return parse_hlp_yaml(out_text)