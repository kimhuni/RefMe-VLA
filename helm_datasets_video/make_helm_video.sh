export PYTHONPATH=$(pwd)
OUT_ROOT="/data/ghkim/helm_data/helm_video_task_10"

# A1
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/press_button_N_times" \
  --out_root="${OUT_ROOT}/press_button_N_times" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/press_button_N_times"


# B1
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/press_button_N_times_M_times_total" \
  --out_root="${OUT_ROOT}/press_button_N_times_M_times_total" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/press_button_N_times_M_times_total"


# A2
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/press_button_in_order" \
  --out_root="${OUT_ROOT}/press_button_in_order" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/press_button_in_order"

# B2
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/press_button_in_human_order" \
  --out_root="${OUT_ROOT}/press_button_in_human_order" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/press_button_in_human_order"

# A3
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/wipe_the_window" \
  --out_root="${OUT_ROOT}/wipe_the_window" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/wipe_the_window"


# B3
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/wipe_the_remaining_window" \
  --out_root="${OUT_ROOT}/wipe_the_remaining_window" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/wipe_the_remaining_window"

# A4
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/find_object_in_drawer" \
  --out_root="${OUT_ROOT}/find_object_in_drawer" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/find_object_in_drawer"

# B4
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/open_drawer_with_object" \
  --out_root="${OUT_ROOT}/open_drawer_with_object" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/open_drawer_with_object"

# A5
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/pick_place_press" \
  --out_root="${OUT_ROOT}/pick_place_press" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/pick_place_press"

# B5
python -m helm_datasets_video.build_videohelm \
  --data_root "/data/ghkim/helm_data/pick_place_original" \
  --out_root="${OUT_ROOT}/pick_place_original" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/pick_place_original"

# Merge
python helm_datasets_video/merge_helm_video_data.py \
  --src_root "/data/ghkim/helm_data/helm_video_task_10" \
  --dst_root "/data/ghkim/helm_data/helm_video_task_10/merged"
