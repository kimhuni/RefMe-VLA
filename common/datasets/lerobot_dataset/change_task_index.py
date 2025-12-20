import pandas as pd
from pathlib import Path
from tqdm import tqdm
from task_config import TASKS_DICT

def change_task_index(data_dir, target_task_index, sample_target):
    for i in sample_target:
        filename = f"episode_{i:06d}.parquet"
        file_path = data_dir / filename
        df = pd.read_parquet(file_path)
        df['task_index'] = target_task_index
        df.to_parquet(file_path, index=False)
        print(file_path, "- to:" ,target_task_index)

if __name__ == "__main__":
    data_dir = Path("/data/ghkim/press the red button/lerobot_5hz/data/chunk-000")
    target_task_index = 0
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/press the green button/lerobot_5hz/data/chunk-000")
    target_task_index = 1
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/press the blue button/lerobot_5hz/data/chunk-000")
    target_task_index = 2
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)