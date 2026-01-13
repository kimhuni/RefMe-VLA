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
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_BGR/data/chunk-000")
    # target_task_index = 23
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_BGR/data/chunk-001")
    # target_task_index = 23
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_BRG/data/chunk-000")
    # target_task_index = 24
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_BRG/data/chunk-001")
    # target_task_index = 24
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_GBR/data/chunk-000")
    # target_task_index = 25
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_GBR/data/chunk-001")
    # target_task_index = 25
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_GRB/data/chunk-000")
    # target_task_index = 26
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_GRB/data/chunk-001")
    # target_task_index = 26
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_RBG/data/chunk-000")
    # target_task_index = 27
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_RBG/data/chunk-001")
    # target_task_index = 27
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_RGB/data/chunk-000")
    # target_task_index = 28
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_in_order/press_button_RGB/data/chunk-001")
    # target_task_index = 28
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)

    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_2/data/chunk-000")
    # target_task_index = 21
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_2/data/chunk-001")
    # target_task_index = 21
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_3/data/chunk-000")
    # target_task_index = 22
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_3/data/chunk-001")
    # target_task_index = 22
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_1/data/chunk-000")
    # target_task_index = 20
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_1/data/chunk-001")
    # target_task_index = 20
    # sample_target = list(range(50, 100))
    # change_task_index(data_dir, target_task_index, sample_target)


    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_2/data/chunk-000")
    # target_task_index = 21
    # sample_target = list(range(0, 50))
    # change_task_index(data_dir, target_task_index, sample_target)



    data_dir = Path("/data/ghkim/lerobot_data/open_the_drawer_ep200/data/chunk-002")
    target_task_index = 8
    sample_target = list(range(100, 150))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/lerobot_data/open_the_drawer_ep200/data/chunk-003")
    target_task_index = 9
    sample_target = list(range(150, 200))
    change_task_index(data_dir, target_task_index, sample_target)



    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_3/data/chunk-000")
    # target_task_index = 32
    # sample_target = list(range(0, 10))
    # change_task_index(data_dir, target_task_index, sample_target)
    #
    #
    # data_dir = Path("/data/ghkim/concat_data/press_button_N_time/press_button_2/data/chunk-000")
    # target_task_index = 33
    # sample_target = list(range(0, 10))
    # change_task_index(data_dir, target_task_index, sample_target)



