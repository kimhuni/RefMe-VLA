export PYTHONPATH=$(pwd)
OUT_ROOT = "/data/ghkim/helm_data/helm_v4_task_8"

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/press_button_N_times" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_N_times" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/press_button_N_times_M_times_total" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_N_times_M_times_total" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/press_button_in_order" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_in_order" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

#python -m helm_datasets_v4.build_helm \
#  --data-root "/data/ghkim/helm_data/press_button_in_human_order" \
#  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
#  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_in_human_order" \
#  --fps_out 5 \
#  --n_images 1 \
#  --val_ratio 0.1 \
#  --shard_size 5000

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/wipe_the_window" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/wipe_the_window" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/wipe_the_remaining_window" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/wipe_the_remaining_window" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/pick_place_press" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/pick_place_press" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

python -m helm_datasets_v4.build_helm \
  --data-root "/data/ghkim/helm_data/pick_place_original" \
  --out_root "/data/ghkim/helm_data/helm_v4_task_8" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/pick_place_original" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# Merge
python helm_datasets_v3/merge_helm_data.py \
  --jsonl_root /data/ghkim/helm_data/helm_v4_task_8/jsonl_v4 \
  --out_dir   /data/ghkim/helm_data/helm_v4_task_8/jsonl_v4/merged_7 \
  --split_mode keep \
  --shard_size 0