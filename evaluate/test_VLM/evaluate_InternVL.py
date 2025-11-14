from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

from PIL import Image, ImageDraw, ImageFont
import os
from textwrap import wrap


torch_device = "cuda"
model_checkpoint = "/ckpt/InternVL3-8B-hf"
processor = AutoProcessor.from_pretrained(
    model_checkpoint
)
model = AutoModelForImageTextToText.from_pretrained(
    model_checkpoint,
    device_map=torch_device,
    dtype=torch.bfloat16
)

SYSTEM_PROMPT = (
    "You are an image analysis expert specialized in robotic manipulation. "
    "You will be given an image showing a robot arm and a text input which consists of robot task and description you generated previously."
    "Describe visible robot actions and task completion strictly based on the image and the input text"
    "Describe in two sentences what the robot is doing and "
    "whether the task is done or not."
)

# SYSTEM_PROMPT = (
#     "You are an expert in robotic manipulation image analysis."
#     "Given: Task description, Previous output (last frame’s action + done status), Current image  "
#
#     "Write exactly two sentences:"
#     "1. Describe what the robot is visibly doing.  "
#     "2. Judge if the task is done, not done, or uncertain."
#
#     "Use only visible evidence from the current image."
#     "Refer to the previous output only for context."
#     "Mark “done” only when physical contact or the final result is clearly seen; otherwise say “not done” or “uncertain.”"
#     "Use short, visual verbs like grasping, pressing, placing, releasing."
#
#     "Task: Press the blue button on the table."
#     "Previous description: The robot's gripper is in contact with the blue button, indicating that it is attempting to press it. The task is not done as the button is still not visibly depressed."
# )

# SYSTEM_PROMPT = (
#     "You are an expert in robotic manipulation image analysis."
#     "You are given:"
#     "1. A task description of what the robot should do."
#     "2. A previous output describing the last frame’s action and task status (frames are 1s apart, so it might be different from current image)."
#     "3. A current image showing the new scene."
#     "Describe what the robot arm is doing in the current image, and judge whether the task is done."
#
#     "Rules: Write exactly two sentences:"
#     "1. What the robot is doing in this image."
#     "2. Whether the task is done, not done, or uncertain."
#     "- Base your judgment only on visible evidence. Use the previous output only for continuity, not as proof."
# #    "- Say “done” only if direct contact or clear result is visible:"
#     "Say “done” only if the gripper is physically pressing or the blue button"
#     "is clearly glowing or deformed due to contact."
#     "If the button’s light is on, assume pressing is complete."
# #    "  - Pressing → gripper visibly pressing or deforming button  "
# #    "  - Placing → object released and stationary at target  "
# #    "  - Grasping → object fully held in closed gripper"
#     "- If evidence is unclear, mark “uncertain.”"
#     "- Use concise visual verbs: moving, grasping, placing, pressing, releasing, returning."
# )

Previous_Description = (
    "The robot arm is positioned above the blue button, appearing to be in the process of pressing it. The task is done."
)

USER_PROMPT = (
    "Task: Press the blue button on the table.\n"
    f"Previous description: f{Previous_Description}"
)

IMAGE_PATH = "/data/piper_press_the_blue_button_screenshot/output_0011.jpg"

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": IMAGE_PATH},
            {"type": "text", "text": USER_PROMPT},
        ],
    }
]

inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)

generate_ids = model.generate(**inputs, max_new_tokens=50)
output_text = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

print(output_text)



# image save

# Create output directory if not exists
output_dir = "/data/piper_press_the_blue_button_screenshot/InternVL"
os.makedirs(output_dir, exist_ok=True)

# Load image
img = Image.open(IMAGE_PATH).convert("RGB")
width, height = img.size

# Extend canvas with a white area below
extra_space = 100
new_img = Image.new("RGB", (width, height + extra_space), (255, 255, 255))
new_img.paste(img, (0, 0))

font_size = 14  # 기본값보다 약 1.5배 큼 (기존 load_default()는 약 11~12px 정도)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)

# Draw text on the white area (with automatic line wrapping)
draw = ImageDraw.Draw(new_img)
text = output_text[0] if isinstance(output_text, list) else str(output_text)

# 자동 줄바꿈: 이미지 폭보다 긴 문장은 wrap()으로 분리
max_width = width - 20  # 여백 고려
lines = []
for line in text.split("\n"):
    lines.extend(wrap(line, width=max(8, int(max_width / max(7, font_size / 2.2)))))  # 글자폭에 맞게 조정 (기본폰트 기준 약 7픽셀/문자)

y_text = height + 10
for line in lines:
    draw.text((10, y_text), line, fill=(0, 0, 0), font=font)
    y_text += 15  # 줄 간격

# Save and notify
save_path = os.path.join(output_dir, os.path.basename(IMAGE_PATH).replace(".jpg", "_InternVL.jpg"))
new_img.save(save_path)
print(f"Annotated image saved to: {save_path}")