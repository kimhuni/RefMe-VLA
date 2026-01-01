import time
import argparse
import logging
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel

"""
python simple_realtime_hlp.py \
  --base_model /path/to/Qwen2.5-VL \
  --adapter /path/to/checkpoint
"""

# ======================================================
# 1. CONFIGURATION (사용자 환경에 맞게 수정)
# ======================================================
# 로봇 카메라 사용 여부 (False면 일반 웹캠/더미 이미지 사용)
USE_ROBOT_CAMERA = False

# 테스트할 고정 명령어 및 메모리 상태
FIXED_TASK = "press the blue button three times"
FIXED_MEMORY = "Action_Command: press the blue button | Working_Memory: Count: 0 (Goal: 3) | Episodic_Context: None"

# 시스템 프롬프트 (templates.py 참조)
DETECT_SYSTEM = (
    "You are the robot arm Visual Event Detector.\n"
    "Goal: Decide whether the target EVENT is detected in the current image.\n"
    "The EVENT corresponds to a meaningful completion moment for the current stage of the Global_Instruction."
    "Input: An image + Global_Instruction describing what counts as action completion"
    " + Memory (may help interpret the current stage/goal)\n"
    "Decision rule:\n"
    "- Use the Global_Instruction  as the primary criterion.\n"
    "- You MAY use Memory only to understand what “completion” means for the current stage."
    "- Event_Detected: true when the completion (or clearly post-completion state) is visible.\n"
    "- Otherwise (partial progress / occlusion / uncertainty) -> Event_Detected: false.\n"
    "Constraints:\n"
    "- Do not propose next actions.\n"
    "- Do not update or rewrite memory.\n"
    "- Do not output any text except YAML.\n"
    "Return YAML with exactly one key: Event_Detected (boolean).\n"
)

logger = logging.getLogger("RealTimeHLP")
logging.basicConfig(level=logging.INFO)


# ======================================================
# 2. MODEL LOADING (eval_helm_hlp_v3.py와 동일)
# ======================================================
def load_model_and_processor(
        base_model: str,
        adapter_path: Optional[str],
        device: str = "cuda:0"
):
    logger.info(f"Loading processor from {base_model}...")
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True)

    # QLoRA Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    logger.info(f"Loading model from {base_model}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # or flash_attention_2
        trust_remote_code=True,
        device_map=device
    )

    if adapter_path:
        logger.info(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


# ======================================================
# 3. CAMERA SETUP (Robot or Webcam)
# ======================================================
class CameraInterface:
    def __init__(self, use_robot: bool = False):
        self.use_robot = use_robot
        self.cap = None
        self.robot_cam = None

        if self.use_robot:
            # ---------------------------------------------------------
            # [참조] eval_real_time_pi0.py의 init_devices 로직
            # 실제 환경에 맞게 import 경로를 확인해주세요.
            # ---------------------------------------------------------
            try:
                from common.utils.utils import init_devices
                from configs.default import DatasetConfig  # Dummy config needed?

                # Mock config object required by init_devices
                class MockConfig:
                    use_devices = True
                    cam_list = ['table']  # wrist 제외 가정

                _, cams = init_devices(MockConfig())
                self.robot_cam = cams['table_rs_cam']
                self.robot_cam.start_recording()
                logger.info("Robot Table Camera Initialized.")
            except ImportError as e:
                logger.error(f"Failed to import robot libraries: {e}")
                logger.warning("Falling back to OpenCV Webcam.")
                self.use_robot = False

        if not self.use_robot:
            self.cap = cv2.VideoCapture(0)  # Webcam 0
            if not self.cap.isOpened():
                logger.warning("No webcam found. Will use black dummy image.")

    def get_image(self) -> Image.Image:
        if self.use_robot and self.robot_cam:
            # eval_real_time_pi0.py: image_for_inference() returns Tensor? or PIL?
            # Assuming it returns a PIL or Tensor that needs conversion.
            # Based on pi0 code: it returns torch tensor usually, check utils.
            # Here we assume we can get a PIL or numpy array.
            # For safety, let's assume we capture a frame:
            img = self.robot_cam.get_latest_frame()  # This method name might vary!
            return Image.fromarray(img)

        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)

        # Fallback Dummy
        return Image.new("RGB", (640, 480), (0, 0, 0))


# ======================================================
# 4. PREPROCESSING (CORE LOGIC FROM eval_helm_hlp_v3)
# ======================================================
def process_inputs_like_offline(
        processor,
        images: List[Image.Image],
        text_prompt: str,
        device: str
) -> Dict[str, torch.Tensor]:
    """
    Offline Eval 코드의 Dataset.__getitem__ 로직을 시뮬레이션합니다.
    특히 image_grid_thw의 shape를 강제로 맞춥니다.
    """

    # 1. Build Messages (Offline style)
    # user_content construction
    content = []
    for _ in range(len(images)):
        content.append({"type": "image"})
    content.append({"type": "text", "text": text_prompt})

    messages = [{"role": "user", "content": content}]

    # 2. Apply Chat Template
    prompt_string = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 3. Processor Call
    # 중요: Offline 코드처럼 padding=False로 처리 후 수동 핸들링하거나,
    # 배치 1개이므로 processor의 출력을 검증.
    model_inputs = processor(
        text=prompt_string,
        images=images,  # Must be a list [img]
        return_tensors="pt",
        padding=False,  # 배치 1이므로 패딩 불필요
    )

    # -------------------------------------------------------
    # [CRITICAL] SHAPE CHECK & CORRECTION (Offline Logic Match)
    # -------------------------------------------------------
    grid_thw = model_inputs.get("image_grid_thw", None)

    print("\n[DEBUG] Raw Processor Output Shapes:")
    for k, v in model_inputs.items():
        if hasattr(v, 'shape'):
            print(f"  - {k}: {v.shape}")

    # Fix grid_thw if needed (eval_helm_hlp_v3 logic)
    if grid_thw is not None:
        # Case 1: (1, 1, 3) -> Squeeze to (1, 3)
        if grid_thw.ndim == 3 and grid_thw.shape[1] == 1:
            grid_thw = grid_thw.squeeze(1)

        # Case 2: (1, 3) is correct for Batch=1.
        # Offline code does: squeeze(0) -> (3,) in dataset -> (1, 3) in collator.
        # Here we just want the final (B, 3) shape which is (1, 3).

        model_inputs["image_grid_thw"] = grid_thw

    # Move to device
    inputs = {k: v.to(device) for k, v in model_inputs.items()}
    return inputs, prompt_string


# ======================================================
# 5. MAIN LOOP
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # 1. Load Model
    model, processor = load_model_and_processor(args.base_model, args.adapter, args.device)

    # 2. Init Camera
    cam = CameraInterface(use_robot=USE_ROBOT_CAMERA)

    logger.info("Ready for Real-time Inference. Press Ctrl+C to stop.")

    step = 0
    try:
        while True:
            # A. Capture
            obs_pil = cam.get_image()

            # 리사이즈 필요시 여기서 수행 (Offline 데이터셋 해상도와 맞춤)
            # obs_pil = obs_pil.resize((MAX_SIZE, MAX_SIZE))

            # B. Prepare Text (Templates.py logic)
            # Images: <image_table> 태그는 processor가 자동 처리하거나,
            # apply_chat_template가 <image> placeholder를 보고 처리함.
            # Offline 데이터셋의 'user_prompt'가 어떻게 구성되었는지에 따라 다름.
            # 여기서는 DETECT_SYSTEM + Task + Memory 조합.

            # NOTE: Offline 코드에서는 user_prompt 안에 이미 <image> 태그나 텍스트가 포함되어 있음.
            # apply_chat_template을 쓰면 <image> 토큰이 자동 삽입됨.
            # 따라서 텍스트 프롬프트에는 이미지 태그를 *텍스트로* 넣을 필요가 없을 수도 있음(Qwen2.5-VL).
            # 하지만 eval_helm_hlp_v3는 user_prompt 텍스트를 그대로 넘김.

            # 수동 구성 (templates.py 참조):
            final_prompt_text = (
                f"{DETECT_SYSTEM}\n"
                f"Task: {FIXED_TASK}\n"
                f"Memory: {FIXED_MEMORY}\n"
                f"Images: <image_table>\n"
            )

            # C. Preprocess (Shape Correction)
            inputs, raw_prompt = process_inputs_like_offline(
                processor,
                [obs_pil],  # List로 감싸는 것이 중요!
                final_prompt_text,
                args.device
            )

            # D. Inference
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False
                )
            dt = time.time() - t0

            # E. Decode
            # remove input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print("=" * 60)
            print(f"[Step {step}] Inference Time: {dt:.3f}s")
            # print(f"[Input Prompt] {raw_prompt[-200:]}...") # 프롬프트 끝부분 확인
            print(f"[Model Output]\n{output_text}")
            print("=" * 60)

            step += 1
            time.sleep(1.0)  # 1초 대기

    except KeyboardInterrupt:
        logger.info("Stopping...")


if __name__ == "__main__":
    main()