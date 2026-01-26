from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# default: Load the model on the available device(s)
local_dir = "/ckpt/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_dir,
    device_map="auto",
    torch_dtype="auto",
    local_files_only=True,
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# IMAGE_PATH = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000034.jpg"
# IMAGE_PATH_2 = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000025.jpg"
# IMAGE_PATH_3 = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000000.jpg"
# IMAGE_PATH_4 = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000008.jpg"
# IMAGE_PATH_5 = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000028.jpg"
# IMAGE_PATH_6 = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000018.jpg"
# IMAGE_PATH_7 = "/data/ghkim/helm_data/find_object_in_drawer/frames_5hz/chunk-002/episode_000100/table/frame_000012.jpg"
#
# USER_PROMPT = "According to the provided images, is there a hamburger in the drawer or is it empty? Explain what is in the drawer"

IMAGE_PATH_1 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000000/table/frame_000039.jpg" # press blue
IMAGE_PATH_2 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000031/table/frame_000034.jpg" # press green
IMAGE_PATH_3 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000046/table/frame_000032.jpg" # press red
IMAGE_PATH_4 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000000/table/frame_000039.jpg" # press blue
IMAGE_PATH_5 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000031/table/frame_000034.jpg" # press green
IMAGE_PATH_6 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000046/table/frame_000032.jpg" # press red
IMAGE_PATH_7 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000000/table/frame_000039.jpg" # press blue
IMAGE_PATH_8 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000031/table/frame_000034.jpg" # press green
IMAGE_PATH_9 = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000046/table/frame_000032.jpg" # press red

IMAGE_PATH_B = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000000/table/frame_000039.jpg" # press blue
IMAGE_PATH_G = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000031/table/frame_000034.jpg" # press green
IMAGE_PATH_R = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000046/table/frame_000032.jpg" # press red

# USER_PROMPT = (
#   "The images are in time order (Image 1,2,3,4,5,6). \n"
#   "For EACH image, exactly identify the color of the button being pressed.\n"
#   "Return one line in this EXACT format:\n"
#   "Final sequence: {image1 : <c1>, image2 : <c2>, image3 : <c3>, image4 : <c4>, image5 : <c5>, image6 : <c6>}\n"
#   "Use only: red, green, blue for the color. No other text."
# )

USER_PROMPT = (
  "The images show a robot arm pressing the blue button.\n"
  "The images are in time order (Image 1,2,3,4). \n"
  "For EACH image, exactly whether the button being pressed.\n"
  "Return one line in this EXACT format:\n"
  "image1 : <c1>, image2 : <c2>, image3 : <c3>, image4 : <c4>\n"
  "Use only: red, blue, green for the color. No other text."
)


# USER_PROMPT = (
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

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE_PATH_1,
            },
            {
                "type": "image",
                "image": IMAGE_PATH_1,
            },
            {
                "type": "image",
                "image": IMAGE_PATH_1,
            },
            {
                "type": "image",
                "image": IMAGE_PATH_1,
            },
            # {
            #     "type": "image",
            #     "image": IMAGE_PATH_5,
            # },
            # {
            #     "type": "image",
            #     "image": IMAGE_PATH_6,
            # },
            # {
            #     "type": "image",
            #     "image": IMAGE_PATH_7,
            # },
            # {
            #     "type": "image",
            #     "image": IMAGE_PATH_8,
            # },
            # {
            #     "type": "image",
            #     "image": IMAGE_PATH_9,
            # },
            {
                "type": "text",
                "text": USER_PROMPT,
            },
        ],
    }
]

# Preparation for inference (run 10 times and measure pure inference time)
import time

times = []

for i in range(5):
    start = time.time()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # print("seq_len:", inputs.input_ids.shape[-1])
    # if hasattr(inputs, "image_grid_thw"):
    #     print("image_grid_thw:", inputs.image_grid_thw)

    # Fix output length to exactly N new tokens (so runtime is comparable across different image counts)
    GEN_NEW_TOKENS = 256

    generated_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=GEN_NEW_TOKENS,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # Actual number of generated tokens (after trimming the prompt)
    print("gen_new_tokens:", generated_ids_trimmed[0].numel())
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)

    print(" - generated text: ", output_text)

    print(f"[Run {i+1}] inference time: {elapsed:.4f} sec--------------------------------------")

# print(f"Average inference time over 10 runs: {sum(times)/len(times):.4f} sec")


# image save
# from PIL import Image, ImageDraw, ImageFont
# import os
# from textwrap import wrap
#
# # Create output directory if not exists
# output_dir = "/data/piper_press_the_blue_button_screenshot/annotated"
# os.makedirs(output_dir, exist_ok=True)
#
# # Load image
# img = Image.open(IMAGE_PATH).convert("RGB")
# width, height = img.size
#
# # Extend canvas with a white area below
# extra_space = 100
# new_img = Image.new("RGB", (width, height + extra_space), (255, 255, 255))
# new_img.paste(img, (0, 0))
#
# font_size = 14  # 기본값보다 약 1.5배 큼 (기존 load_default()는 약 11~12px 정도)
# font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
#
# # Draw text on the white area (with automatic line wrapping)
# draw = ImageDraw.Draw(new_img)
# text = output_text[0] if isinstance(output_text, list) else str(output_text)
#
# # 자동 줄바꿈: 이미지 폭보다 긴 문장은 wrap()으로 분리
# max_width = width - 20  # 여백 고려
# lines = []
# for line in text.split("\n"):
#     lines.extend(wrap(line, width=int(max_width / 7)))  # 글자폭에 맞게 조정 (기본폰트 기준 약 7픽셀/문자)
#
# y_text = height + 10
# for line in lines:
#     draw.text((10, y_text), line, fill=(0, 0, 0), font=font)
#     y_text += 15  # 줄 간격
#
# # Save and notify
# save_path = os.path.join(output_dir, os.path.basename(IMAGE_PATH).replace(".jpg", "_test_simple.jpg"))
# new_img.save(save_path)
# print(f"Annotated image saved to: {save_path}")
