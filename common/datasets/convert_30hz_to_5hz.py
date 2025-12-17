import os
from tqdm import tqdm
import cv2
from datasets import load_dataset, Dataset, Features, Sequence, Value

def cvt_vid(origin, output):
    cap = cv2.VideoCapture(origin)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 비디오 설정 (5Hz로 저장)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, 5, (width, height))  # fps=5

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 6 프레임마다 하나 저장
        if frame_idx % 6 == 0:
            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()

def convert_30hz_to_5hz(source_base_path, dest_base_path, index, task_index):
    # 반복되는 이름들을 변수로 선언
    chunk_dir_name = f"chunk-{index // 50:03d}"
    episode_name = f"episode_{index:06d}"

    # 참고: 원본 코드의 파일 경로에 'train'이 있어 폴더 생성 시에도 추가했습니다.
    os.makedirs(f"{dest_base_path}/data/{chunk_dir_name}", exist_ok=True)
    os.makedirs(f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.exo", exist_ok=True)
    os.makedirs(f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.wrist", exist_ok=True)
    os.makedirs(f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.table", exist_ok=True)

    # 3. 원본 파일 경로를 동적으로 생성
    parquet_file_path = f"{source_base_path}/data/{chunk_dir_name}/{episode_name}.parquet"
    exo_video_file_path = f"{source_base_path}/videos/{chunk_dir_name}/observation.images.exo/{episode_name}.mp4"
    wrist_video_file_path = f"{source_base_path}/videos/{chunk_dir_name}/observation.images.wrist/{episode_name}.mp4"
    table_video_file_path = f"{source_base_path}/videos/{chunk_dir_name}/observation.images.table/{episode_name}.mp4"

    # 4. 대상 파일 경로를 동적으로 생성
    parquet_file_path_des = f"{dest_base_path}/data/{chunk_dir_name}/{episode_name}.parquet"
    exo_video_file_path_des = f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.exo/{episode_name}.mp4"
    wrist_video_file_path_des = f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.wrist/{episode_name}.mp4"
    table_video_file_path_des = f"{dest_base_path}/videos/{chunk_dir_name}/observation.images.table/{episode_name}.mp4"


    dataset = load_dataset("parquet", data_files=parquet_file_path)['train']
    features = Features({
        "action": Sequence(Value("int64"), length=7),
        "action_joint": Sequence(Value("int64"), length=7),
        "observation.state": Sequence(Value("int64"), length=7),
        "observation.state_joint": Sequence(Value("int64"), length=7),
        "timestamp": Value("float32"),
        "frame_index": Value("int64"),
        "episode_index": Value("int64"),
        "task_index": Value("int64")
    })

    dataset = dataset.to_dict()

    dataset['action'] = [elem[0] for elem in dataset['action']]
    dataset['action_joint'] = [elem[0] for elem in dataset['action_joint']]
    dataset['observation.state'] = [elem[0] for elem in dataset['observation.state']]
    dataset['observation.state_joint'] = [elem[0] for elem in dataset['observation.state_joint']]
    dataset['task_index'] = [task_index for _ in dataset['task_index']]
    # dataset['task_index'] = [index for _ in dataset['task_index']]

    dataset = Dataset.from_dict(dataset, features=features)
    sampled_dataset = dataset.select(range(0, len(dataset), 6))
    sampled_dataset.to_parquet(parquet_file_path_des)

    cvt_vid(exo_video_file_path, exo_video_file_path_des)
    cvt_vid(wrist_video_file_path, wrist_video_file_path_des)
    cvt_vid(table_video_file_path, table_video_file_path_des)


if __name__ == "__main__":
    # for i in tqdm(range(0,20)):
    #     convert_30hz_to_5hz("/data/ghkim/wipe_the_window/lerobot", "/data/ghkim/wipe_the_window/lerobot_5hz", i)
    for i in tqdm(range(0, 20)):
        convert_30hz_to_5hz("/data/ghkim/press the green button/lerobot",
                            "/data/ghkim/press the green button/lerobot_5hz",
                            i,1)

    for i in tqdm(range(0, 20)):
        convert_30hz_to_5hz("/data/ghkim/press the blue button/lerobot",
                            "/data/ghkim/press the blue button/lerobot_5hz",
                            i,2)

    for i in tqdm(range(0, 20)):
        convert_30hz_to_5hz("/data/ghkim/press the red button/lerobot",
                            "/data/ghkim/press the red button/lerobot_5hz",
                            i,0)

    # convert_30hz_to_5hz("/data/piper_open_the_pot_0804_ep120/lerobot", "/data/piper_open_the_pot_0804_ep120/lerobot_5hz", 119)
    # for i in tqdm(range(200,1200)):
    #     convert_30hz_to_5hz("/data/piper_corn_grape_0717/lerobot_rearranged", "/data/piper_corn_grape_0717/lerobot_5hz_re", i)