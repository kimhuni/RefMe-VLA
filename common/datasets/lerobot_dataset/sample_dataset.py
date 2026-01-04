import os
from tqdm import tqdm
import shutil
import random

from datasets import load_dataset

def sample_dataset(sample_target:list[int], data_dir, target_dir, start_idx):
    for index, target_index in tqdm(enumerate(sample_target), total=len(sample_target)):
        index += start_idx
        os.makedirs(f"{target_dir}/data/chunk-{index // 50:03d}", exist_ok=True)
        os.makedirs(
            f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.exo",
            exist_ok=True)
        os.makedirs(
            f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.wrist",
            exist_ok=True)
        os.makedirs(
            f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.table",
            exist_ok=True)

        parquet_src_file = f"{data_dir}/data/chunk-{target_index // 50:03d}/episode_{target_index:06d}.parquet"
        parquet_des_file = f"{target_dir}/data/chunk-{index // 50:03d}/episode_{index:06d}.parquet"

        exo_src_file = f"{data_dir}/videos/chunk-{target_index // 50:03d}/observation.images.exo/episode_{target_index:06d}.mp4"
        exo_des_file = f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.exo/episode_{index:06d}.mp4"

        wrist_src_file = f"{data_dir}/videos/chunk-{target_index // 50:03d}/observation.images.wrist/episode_{target_index:06d}.mp4"
        wrist_des_file = f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.wrist/episode_{index:06d}.mp4"

        table_src_file = f"{data_dir}/videos/chunk-{target_index // 50:03d}/observation.images.table/episode_{target_index:06d}.mp4"
        table_des_file = f"{target_dir}/videos/chunk-{index // 50:03d}/observation.images.table/episode_{index:06d}.mp4"

        src_dataset = load_dataset("parquet", data_files=parquet_src_file)['train']
        des_dataset = src_dataset.map(lambda x: {"episode_index": index})
        des_dataset.to_parquet(parquet_des_file)

        shutil.copy(exo_src_file, exo_des_file)
        shutil.copy(wrist_src_file, wrist_des_file)
        shutil.copy(table_src_file, table_des_file)


import random


def generate_unique_random_numbers_in_intervals(start, end, interval=10, num_per_interval=2):
    """
    지정된 시작부터 끝 범위까지 주어진 간격마다 겹치지 않는 랜덤 숫자를 지정된 개수만큼 뽑아 리스트로 반환합니다.
    """
    all_sampled_numbers = []

    current_start = start

    while current_start <= end:
        current_end = min(current_start + interval - 1, end)

        # 현재 간격 내에서 뽑을 수 있는 모든 숫자 후보군
        possible_numbers_in_interval = list(range(current_start, current_end + 1))

        # 만약 현재 간격의 길이가 뽑으려는 숫자 개수보다 작으면,
        # 해당 간격에서 뽑을 수 있는 최대 개수만큼만 뽑습니다.
        numbers_to_sample = min(len(possible_numbers_in_interval), num_per_interval)

        # 유효한 숫자가 있는 경우에만 샘플링
        if numbers_to_sample > 0:
            # random.sample을 사용하여 겹치지 않는 숫자들을 뽑습니다.
            sampled_numbers_in_current_interval = random.sample(possible_numbers_in_interval, numbers_to_sample)
            # 현재 구간에서 뽑힌 숫자들을 최종 리스트에 추가합니다.
            all_sampled_numbers.extend(sampled_numbers_in_current_interval)

        current_start += interval

    all_sampled_numbers.sort()
    return all_sampled_numbers



if __name__ == '__main__':

    # print(sample_target)

    # which episode you want to sample
    sample_target = list(range(0, 20))
    # or random sample
    # sample_target = generate_unique_random_numbers_in_intervals(0, 2759, 2160, 600)

    # directory to merge
    target_dir = '/data/ghkim/open_the_drawer'

    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'data'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'meta'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'videos'), exist_ok=True)

    # merge dataset 1
    data_dir = '/data/ghkim/open_the_drawer/open_leftdown_drawer/lerobot_5hz'
    sample_dataset(sample_target, data_dir, target_dir, 0)

    # merge dataset 2
    data_dir = '/data/ghkim/open_the_drawer/open_leftup_drawer/lerobot_5hz'
    sample_dataset(sample_target, data_dir, target_dir, 20)

    # merge dataset 3
    data_dir = '/data/ghkim/open_the_drawer/open_rightdown_drawer/lerobot_5hz'
    sample_dataset(sample_target, data_dir, target_dir, 40)

    data_dir = '/data/ghkim/open_the_drawer/open_rightup_drawer/lerobot_5hz'
    sample_dataset(sample_target, data_dir, target_dir, 60)

    print(f"merged at {sample_target}")


    # data_dir = '/data/piper_pour/lerobot_5hz'
    # os.makedirs(target_dir, exist_ok=True)
    # os.makedirs(os.path.join(target_dir, 'data'), exist_ok=True)
    # os.makedirs(os.path.join(target_dir, 'meta'), exist_ok=True)
    # os.makedirs(os.path.join(target_dir, 'videos'), exist_ok=True)
    #
    # sample_dataset(sample_target, data_dir, target_dir, 360)