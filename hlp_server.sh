# HLP 전용 env에서 서버 실행
python evaluate/eval_real_time_API/hlp_server.py \
  --adapter_dir /home/minji/Desktop/data/finetuned_model/ghkim/HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1114_final/checkpoint-2000 \
  --base_dir /home/minji/Desktop/data/ckpt/Qwen2.5-VL-7B-Instruct \
  --use_4bit \
  --host 127.0.0.1 --port 8787