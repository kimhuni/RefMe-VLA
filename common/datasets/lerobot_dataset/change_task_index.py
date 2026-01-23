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
    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_bottom_left/data/chunk-000")
    target_task_index = 40
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_bottom_left/data/chunk-001")
    target_task_index = 40
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_top_left/data/chunk-000")
    target_task_index = 41
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_top_left/data/chunk-001")
    target_task_index = 41
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_bottom_right/data/chunk-000")
    target_task_index = 42
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_bottom_right/data/chunk-001")
    target_task_index = 42
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_top_right/data/chunk-000")
    target_task_index = 43
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/find_object_in_drawer/find_top_right/data/chunk-001")
    target_task_index = 43
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)
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


# BWR
    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_BWR/data/chunk-000")
    target_task_index = 34
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_BWR/data/chunk-001")
    target_task_index = 34
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    # BWR
    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_BRW/data/chunk-000")
    target_task_index = 35
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_BRW/data/chunk-001")
    target_task_index = 35
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    # BWR
    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_RBW/data/chunk-000")
    target_task_index = 36
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_RBW/data/chunk-001")
    target_task_index = 36
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    # BWR
    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_RWB/data/chunk-000")
    target_task_index = 37
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_RWB/data/chunk-001")
    target_task_index = 37
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    # BWR
    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_WBR/data/chunk-000")
    target_task_index = 38
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_WBR/data/chunk-001")
    target_task_index = 38
    sample_target = list(range(50, 100))
    change_task_index(data_dir, target_task_index, sample_target)

    # BWR
    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_WRB/data/chunk-000")
    target_task_index = 39
    sample_target = list(range(0, 50))
    change_task_index(data_dir, target_task_index, sample_target)

    data_dir = Path("/data/ghkim/concat_data/pick_place_press/pick_place_press_WRB/data/chunk-001")
    target_task_index = 39
    sample_target = list(range(50, 100))
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



