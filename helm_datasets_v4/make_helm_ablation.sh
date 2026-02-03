export PYTHONPATH=$(pwd)
OUT_ROOT="/data/ghkim/helm_data/helm_ablation_task_5"

# A1
python -m helm_datasets_v4.build_helm_ablation \
  --data_root "/data/ghkim/helm_data/press_button_in_order" \
  --out_root "${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_N_times" \
  --shard_size 5000

# A2
python -m helm_datasets_v4.build_helm_ablation \
  --data_root "/data/ghkim/helm_data/press_button_in_order" \
  --out_root "${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/press_button_in_order" \
  --shard_size 5000

# A3
python -m helm_datasets_v4.build_helm_ablation \
  --data_root "/data/ghkim/helm_data/wipe_the_window" \
  --out_root "${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/wipe_the_window" \
  --shard_size 5000

# A4
python -m helm_datasets_v4.build_helm_ablation \
  --data_root "/data/ghkim/helm_data/find_object_in_drawer" \
  --out_root "${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/find_object_in_drawer" \
  --shard_size 5000

# A5
python -m helm_datasets_v4.build_helm_ablation \
  --data_root "/data/ghkim/helm_data/pick_place_press" \
  --out_root "${OUT_ROOT}" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v4/taskspecs/pick_place_press" \
  --shard_size 5000

# Merge
python helm_datasets_v4/merge_helm_data.py \
  --jsonl_root "${OUT_ROOT}/jsonl_v4" \
  --out_dir   "${OUT_ROOT}/merged" \
  --split_mode keep \
  --shard_size 0