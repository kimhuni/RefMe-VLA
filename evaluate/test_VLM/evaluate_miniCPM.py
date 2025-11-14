import os
import torch
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap
from transformers import AutoModel, AutoTokenizer

torch.manual_seed(100)
model_checkpoint = "/ckpt/MiniCPM-V-4_5"

model = AutoModel.from_pretrained(
    model_checkpoint,
    trust_remote_code=True, # or openbmb/MiniCPM-o-2_6
    attn_implementation='sdpa',
    dtype=torch.bfloat16
) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint,
    trust_remote_code=True
)


enable_thinking=False # If `enable_thinking=True`, the thinking mode is enabled.
stream=True # If `stream=True`, the answer is string


SYSTEM_PROMPT = (
    "You are an image analysis expert specialized in robotic manipulation. "
    "You will be given an image showing a robot arm and a text input which consists of robot task and description you generated previously."
    "Describe visible robot actions and task completion strictly based on the image and the input text"
    "Describe in two sentences what the robot is doing and "
    "whether the task is done or not."
)

image = Image.open("/result/VLM_test/piper_press_the_blue_button_screenshot/output_0010.jpg").convert('RGB')

Previous_Description = (
    "The robot arm is positioned above the blue button, ready to press it. The task is not done."
)

USER_PROMPT = (
    "Task: Press the blue button on the table.\n"
    f"Previous description: f{Previous_Description}"
)

IMAGE_PATH = "/result/VLM_test/piper_press_the_blue_button_screenshot/output_0010.jpg"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
        ],
    },
    {
        "role": "user",
        "content": [image, USER_PROMPT],
    }
]

answer = model.chat(
    msgs=messages,
    tokenizer=tokenizer,
    enable_thinking=enable_thinking,
    stream=stream
)

# Collect streamed text if streaming, otherwise use the direct string
if stream:
    output_text = ""
    for new_text in answer:
        output_text += new_text
        print(new_text, flush=True, end="")
    print()  # newline after streaming
else:
    output_text = answer

# generated_text = ""
# for new_text in answer:
#     generated_text += new_text
#     print(new_text, flush=True, end='')

# Second round chat, pass history context of multi-turn conversation
# msgs.append({"role": "assistant", "content": [generated_text]})
# msgs.append({"role": "user", "content": ["What should I pay attention to when traveling here?"]})
#
# answer = model.chat(
#     msgs=msgs,
#     tokenizer=tokenizer,
#     stream=True
# )
#
# generated_text = ""
# for new_text in answer:
#     generated_text += new_text
#     print(new_text, flush=True, end='')

#################################################################################

# image save

# Create output directory if not exists
output_dir = "/result/VLM_test/piper_press_the_blue_button_screenshot/MiniCPM"
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
save_path = os.path.join(output_dir, os.path.basename(IMAGE_PATH).replace(".jpg", "_MiniCPM.jpg"))
new_img.save(save_path)
print(f"Annotated image saved to: {save_path}")