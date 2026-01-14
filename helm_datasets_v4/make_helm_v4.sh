export PYTHONPATH=$(pwd)
OUT_ROOT="/data/ghkim/helm_data/helm_v4_task_10"

# A1
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/press_button_N_times" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_N_times" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# B1
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/press_button_N_times_M_times_total" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_N_times_M_times_total" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# A2
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/press_button_in_order" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_in_order" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# B2
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/press_button_in_human_order" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_in_human_order" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# A3
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/wipe_the_window" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/wipe_the_window" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# B3
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/wipe_the_remaining_window" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/wipe_the_remaining_window" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# A4
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/find_object_in_drawer" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/find_object_in_drawer" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# B4
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/open_drawer_with_object" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/open_drawer_with_object" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# A5
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/pick_place_press" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/pick_place_press" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# B5
python -m helm_datasets_v4.build_helm \
  --data_root "/data/ghkim/helm_data/pick_place_original" \
  --out_root="${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/pick_place_original" \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000

# Merge
python helm_datasets_v4/merge_helm_data.py \
  --jsonl_root "${OUT_ROOT}/jsonl_v4" \
  --out_dir   "${OUT_ROOT}/merged" \
  --split_mode keep \
  --shard_size 0