LLP_BASE_MODEL="/home/minji/Desktop/data/finetuned_model/ghkim/pi0_press_the_blue_button_ep60/030000/pretrained_model"
LLP_DATASET="/home/minji/Desktop/data/data_config/ep60_press_the_blue_button"

python -m evaluate.eval_real_time_API.eval_real_time_API_main \
  --use_hlp 1 \
  --use_remote_hlp 1 \
  --hlp_url http://127.0.0.1:8787 \
  --hlp_period 5 \
  --task "press the blue button" \
  --max_steps 200 \
  --use_devices 1 \
  --llp_device "cuda" \
  --llp_model_path "${HLP_BASE_MODEL}" \
  --dataset_repo_id "${LLP_DATASET}" \
  --dataset_root "${LLP_DATASET}"