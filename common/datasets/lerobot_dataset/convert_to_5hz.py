import os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

def cvt_vid(origin, output, target_fps=5):
    cap = cv2.VideoCapture(origin)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_fps is None or src_fps <= 0:
        raise ValueError(f"Invalid source FPS from video: {origin} -> {src_fps}")

    # Compute stride: how many source frames to skip per one output frame.
    stride = int(round(float(src_fps) / float(target_fps)))
    if stride < 1:
        stride = 1

    # 출력 비디오 설정 (target_fps로 저장)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, target_fps, (width, height))  # fps=target_fps

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # stride 프레임마다 하나 저장
        if frame_idx % stride == 0:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

    return stride, src_fps

def convert_to_5hz(source_base_path, dest_base_path, index, task_index=None, target_fps=5):
    # 반복되는 이름들을 변수로 선언
    chunk_dir_name = f"chunk-{index // 1000:03d}"
    episode_name = f"episode_{index:06d}"

    # 참고: 원본 코드의 파일 경로에 'train'이 있어 폴더 생성 시에도 추가했습니다.
    os.makedirs(f"{dest_base_path}/data/{chunk_dir_name}", exist_ok=True)
    # os.makedirs(f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.exo", exist_ok=True)
    os.makedirs(f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.wrist", exist_ok=True)
    os.makedirs(f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.table", exist_ok=True)

    # 3. 원본 파일 경로를 동적으로 생성
    parquet_file_path = f"{source_base_path}/data/{chunk_dir_name}/{episode_name}.parquet"
    # exo_video_file_path = f"{source_base_path}/videos/{chunk_dir_name}/observation.images.exo/{episode_name}.mp4"
    wrist_video_file_path = f"{source_base_path}/videos/{chunk_dir_name}/observation.images.wrist/{episode_name}.mp4"
    table_video_file_path = f"{source_base_path}/videos/{chunk_dir_name}/observation.images.table/{episode_name}.mp4"

    # 4. 대상 파일 경로를 동적으로 생성
    parquet_file_path_des = f"{dest_base_path}/data/{chunk_dir_name}/{episode_name}.parquet"
    # exo_video_file_path_des = f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.exo/{episode_name}.mp4"
    wrist_video_file_path_des = f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.wrist/{episode_name}.mp4"
    table_video_file_path_des = f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.table/{episode_name}.mp4"

    # ---- Load parquet with pandas (robust for list/ndarray columns) ----
    df = pd.read_parquet(parquet_file_path)

    if task_index is not None and "task_index" in df.columns:
        df["task_index"] = int(task_index)

    # Determine stride from the actual video FPS (table camera) so parquet/video stay in sync.
    table_cap = cv2.VideoCapture(table_video_file_path)
    src_fps = table_cap.get(cv2.CAP_PROP_FPS)
    table_cap.release()
    if src_fps is None or src_fps <= 0:
        raise ValueError(f"Invalid source FPS from video: {table_video_file_path} -> {src_fps}")

    stride = int(round(float(src_fps) / float(target_fps)))
    if stride < 1:
        stride = 1

    # ---- Resample dataframe ----
    idxs = list(range(0, len(df), stride))
    df_s = df.iloc[idxs].reset_index(drop=True)

    n = len(df_s)
    if "frame_index" in df_s.columns:
        df_s["frame_index"] = np.arange(n, dtype=np.int64)
    if "timestamp" in df_s.columns:
        df_s["timestamp"] = (np.arange(n, dtype=np.float32) / np.float32(target_fps)).astype(np.float32)
    if "episode_index" in df_s.columns:
        df_s["episode_index"] = np.full(n, int(index), dtype=np.int64)
    if task_index is not None and "task_index" in df_s.columns:
        df_s["task_index"] = np.full(n, int(task_index), dtype=np.int64)
    if "index" in df_s.columns:
        df_s["index"] = np.arange(n, dtype=np.int64)

    df_s.to_parquet(parquet_file_path_des, index=False)

    # cvt_vid(exo_video_file_path, exo_video_file_path_des, target_fps=target_fps)
    cvt_vid(wrist_video_file_path, wrist_video_file_path_des, target_fps=target_fps)
    cvt_vid(table_video_file_path, table_video_file_path_des, target_fps=target_fps)


if __name__ == "__main__":
    # for i in tqdm(range(0,20)):
    #     convert_30hz_to_5hz("/data/ghkim/wipe_the_window/lerobot", "/data/ghkim/wipe_the_window/lerobot_5hz", i)
    # for i in tqdm(range(0, 20)):
    #     convert_30hz_to_5hz("/data/ghkim/pick_place_press/banana_blue_to_red/lerobot",
    #                         "/data/ghkim/pick_place_press/banana_blue_to_red/lerobot_5hz",
    #                         i,10)
    #
    # for i in tqdm(range(0, 20)):
    #     convert_30hz_to_5hz("/data/ghkim/pick_place_press/banana_blue_to_white/lerobot",
    #                         "/data/ghkim/pick_place_press/banana_blue_to_white/lerobot_5hz",
    #                         i,11)
    #
    # for i in tqdm(range(0, 20)):
    #     convert_30hz_to_5hz("/data/ghkim/pick_place_press/banana_red_to_blue/lerobot",
    #                         "/data/ghkim/pick_place_press/banana_red_to_blue/lerobot_5hz",
    #                         i,12)
    #
    # for i in tqdm(range(0, 20)):
    #     convert_30hz_to_5hz("/data/ghkim/pick_place_press/banna_red_to_white/lerobot",
    #                         "/data/ghkim/pick_place_press/banana_red_to_white/lerobot_5hz",
    #                         i,13)
    #
    # for i in tqdm(range(0, 20)):
    #     convert_30hz_to_5hz("/data/ghkim/pick_place_press/banana_white_to_blue/lerobot",
    #                         "/data/ghkim/pick_place_press/banana_white_to_blue/lerobot_5hz",
    #                         i,14)

    # for i in tqdm(range(0, 10)):
    #     convert_30hz_to_5hz("/data/ghkim/data_hub/open_empty_drawer_ep40/open_leftdown_drawer_empty/lerobot",
    #                         "/data/ghkim/data_hub/open_empty_drawer_ep40/open_leftdown_drawer_empty/lerobot_5hz",
    #                         i, 30)
    #
    # for i in tqdm(range(0, 10)):
    #     convert_30hz_to_5hz("/data/ghkim/data_hub/open_empty_drawer_ep40/open_leftup_drawer_empty/lerobot",
    #                         "/data/ghkim/data_hub/open_empty_drawer_ep40/open_leftup_drawer_empty/lerobot_5hz",
    #                         i,31)
    #
    # for i in tqdm(range(0, 10)):
    #     convert_30hz_to_5hz("/data/ghkim/data_hub/open_empty_drawer_ep40/open_rightdown_drawer_empty/lerobot",
    #                         "/data/ghkim/data_hub/open_empty_drawer_ep40/open_rightdown_drawer_empty/lerobot_5hz",
    #                         i,32)
    #
    # for i in tqdm(range(0, 10)):
    #     convert_30hz_to_5hz("/data/ghkim/data_hub/open_empty_drawer_ep40/open_rightup_drawer_empty/lerobot",
    #                          "/data/ghkim/data_hub/open_empty_drawer_ep40/open_rightup_drawer_empty/lerobot_5hz",
    #                         i, 33)

    for i in tqdm(range(0,960)):
        convert_to_5hz("/data/libero-mem_lerobot", "/data/libero-mem_lerobot_5hz", i)