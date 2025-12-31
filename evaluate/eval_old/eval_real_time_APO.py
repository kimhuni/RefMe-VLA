import time
import logging
from pprint import pformat
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
from termcolor import colored
import torch
import safetensors.torch as sft
import copy

from common.constants import GRIPPER_EFFORT
from common.robot_devices.robot_utils import read_end_pose_msg, ctrl_end_pose, read_joint_msg, set_zero_configuration
from common.utils.utils import (
    load_buffer,
    get_current_action,
    random_piper_action,
    random_piper_image,
    plot_trajectory,
    pretty_plot,
    log_time,
    init_devices
)
from configs.eval_real_time_ours_APO import EvalRealTimeOursPipelineConfig

from common.utils.random_utils import set_seed
from common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
    init_keyboard_listener
)
from configs import parser

from common.policies.APO.configuration_APO import APOConfig
from common.policies.APO.modeling_APO import APO

from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.datasets.utils import dataset_to_policy_features
from configs.types import FeatureType

def create_batch(piper, wrist_rs_cam, use_devices, task, use_end_pose: bool = True):
    if use_devices:
        return {
            'observation.state': read_end_pose_msg(piper) if use_end_pose else read_joint_msg(piper),
            'observation.images.wrist': wrist_rs_cam.image_for_inference(),
            'task': [task],
        }
    else:
        return {
            'observation.state': random_piper_action(),
            'observation.images.wrist': random_piper_image(),
            'task': [task],
        }


@parser.wrap()
def eval_real_time(cfg: EvalRealTimeOursPipelineConfig):
    ###############
    # INIT DEVICES
    ###############
    if cfg.use_devices:
        piper, cam = init_devices(cfg)
        wrist_rs_cam = cam['wrist_rs_cam']
        exo_rs_cam = cam['exo_rs_cam']
        table_rs_cam = cam['table_rs_cam']

        listener, event, task = init_keyboard_listener()

    else:
        piper = None
        wrist_rs_cam = None
        exo_rs_cam = None
        table_rs_cam = None

        listener, event, task = None, None, None

    logging.info(pformat(cfg.to_dict()))
    #cfg.target_keywords = ['q_proj', 'k_proj', 'v_proj']

    device = "cuda"
    torch.backends.cudnn.benchmark = True

    if cfg.seed is not None:
        set_seed(cfg.seed)


    ###############
    # LOAD DATASET
    ###############
    logging.info("Creating dataset")
    dataset_metadata = LeRobotDatasetMetadata(cfg.dataset_path)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    policy_cfg = APOConfig(input_features=input_features, output_features=output_features)

    logging.info("Making policy.")
    policy = APO.from_pretrained(
        cfg.policy_path,
        config=policy_cfg,
        dataset_stats=dataset_metadata.stats,
        device=device,
    )

    policy.to(device)
    policy.visual_proj.to(device)
    policy.vision_encoder.to(device)
    step = 0
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        table_rs_cam.start_recording()
        logging.info("Devices started recording")

    policy.eval()

    logging.info("Start offline evaluation on a fixed dataset")

    buffer = [[] for _ in range(policy.config.n_action_steps)]
    action_pred_list = []

    set_zero_configuration(piper)
    ###############
    # EVAL LOOP
    ###############
    while True:
        t_start = log_time()

        # emergency stop
        if cfg.use_devices and event["stop recording"]:
            set_zero_configuration(piper)
            time.sleep(1)
            logging.info('EMERGENCY STOP... RESTARTING...')
            event['stop recording'] = False
            continue

        if cfg.use_devices and task['task1 : open the pot']:
            set_zero_configuration(piper)

            stt = read_end_pose_msg(piper)
            end_pose_data = stt[0][:6].tolist()
            gripper_data = [torch.tensor(60000), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            print('gripper open')

            time.sleep(3)
            cfg.task = "<action> open the pot."
            logging.info(cfg.task)
            task['task1 : open the pot'] = False
            continue
        if cfg.use_devices and task['task2 : pour the block']:
            set_zero_configuration(piper)

            stt = read_end_pose_msg(piper)
            end_pose_data = stt[0][:6].tolist()
            gripper_data = [torch.tensor(60000), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            print('gripper open')

            time.sleep(3)
            # cfg.task = "pour the block into the basket"
            cfg.task = "<action> pour the black ball from the white bowl into the basket."
            logging.info(cfg.task)
            task['task2 : pour the block'] = False
            continue
        if cfg.use_devices and task['task3 : push the button']:
            set_zero_configuration(piper)
            time.sleep(3)
            cfg.task = "<action> push the button."
            logging.info(cfg.task)
            task['task3 : push the button'] = False
            continue
        if cfg.use_devices and task['task4 : pick and place']:
            set_zero_configuration(piper)

            stt = read_end_pose_msg(piper)
            end_pose_data = stt[0][:6].tolist()
            gripper_data = [torch.tensor(60000), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            print('gripper open')

            time.sleep(3)
            cfg.task = "<action> pick and place the grape in the basket."
            logging.info(cfg.task)
            task['task4 : pick and place'] = False
            continue

            #
        # create batch
        print(cfg.task)

        # create batch
        batch = create_batch(piper, wrist_rs_cam, cfg.use_devices, cfg.task)

        t_create_batch = log_time()

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        t_batch_to_gpu = log_time()

        # infer data
        action_pred = policy.inference_action(batch).squeeze()


        t_action_pred = log_time()

        for i in range(10):
            action_chunk = action_pred[i]
            # actuate robot
            end_pose_data = action_chunk[:6].cpu().to(dtype=int).tolist()
            gripper_data = [action_chunk[6].cpu().to(dtype=int), GRIPPER_EFFORT]
            ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
            t_action_publish = log_time()
            print(f'action_pred: {action_chunk}')
            time.sleep(0.2)

        step += 1


        t_total = log_time()
        # logged_time = logged_time | {
        #     "action_pred": action_pred,
        #     "t_create_batch": t_create_batch - t_start,
        #     "t_batch_to_gpu": t_batch_to_gpu - t_create_batch,
        #     "t_action_pred": t_action_pred - t_batch_to_gpu,
        #     "t_action_publish": t_action_publish - t_action_pred,
        #     "t_total": t_total - t_start,
        # }
        # logging.info(colored(pformat(logged_time), "yellow", attrs=["bold"]))

        if step > cfg.max_steps:
            break
        pass


if __name__ == "__main__":
    init_logging()
    eval_real_time()