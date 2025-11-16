# realtime_LLP_pi0.py

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import logging
import time

import torch
from termcolor import colored

from lerobot.common.configs import parser
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    set_seed,
)
from lerobot.scripts.eval_real_time_utils import (
    init_devices,
    read_end_pose_msg,
    read_joint_msg,
    random_piper_action,
    random_piper_image,
    ctrl_end_pose,
    set_zero_configuration,
    init_keyboard_listener,
)
from lerobot.common.policies.factory import make_policy, wrap_policy

# 여기는 프로젝트에서 실제 사용하는 Config 경로/이름으로 맞춰줘야 함
from lerobot.common.configs.policy import PolicyConfig  # 예시
from lerobot.common.configs.method import MethodConfig  # 예시
from lerobot.common.configs.dataset import DatasetConfig  # 예시

GRIPPER_EFFORT = 10000  # 프로젝트 설정에 맞게 수정


@dataclass
class LLPConfig:
    use_devices: bool = True
    task: str = "press the blue button"
    max_steps: int = 1000
    seed: Optional[int] = None

    # LLP 모델 관련
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    train_dataset: DatasetConfig = field(default_factory=DatasetConfig)


def create_llp_batch(
    piper,
    table_rs_cam,
    wrist_rs_cam,
    use_devices: bool,
    task: str,
    use_end_pose: bool = True,
) -> Dict[str, Any]:
    if use_devices:
        return {
            "observation.state": read_end_pose_msg(piper) if use_end_pose else read_joint_msg(piper),
            "observation.images.table": table_rs_cam.image_for_inference(),
            "observation.images.wrist": wrist_rs_cam.image_for_inference(),
            "task": [task],
        }
    else:
        return {
            "observation.state": random_piper_action(),
            "observation.images.table": random_piper_image(),
            "observation.images.wrist": random_piper_image(),
            "task": [task],
        }


@dataclass
class LLPRuntimeContext:
    """LLP가 돌아가기 위해 필요한 runtime 리소스 묶음."""
    cfg: LLPConfig
    device: torch.device
    policy: torch.nn.Module
    piper: Any
    table_rs_cam: Any
    wrist_rs_cam: Any
    exo_rs_cam: Any
    keyboard_event: Dict[str, Any]


def init_llp_runtime(cfg: LLPConfig) -> LLPRuntimeContext:
    # 1) 디바이스 / 시드
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 2) 로봇 / 카메라 / 키보드
    if cfg.use_devices:
        piper, cam = init_devices(cfg)
        wrist_rs_cam = cam["wrist_rs_cam"]
        exo_rs_cam = cam["exo_rs_cam"]
        table_rs_cam = cam["table_rs_cam"]

        listener, event, _task_state = init_keyboard_listener()
        # 여기서 ESC → event["reset_hlp"]=True 같은 매핑은
        # init_keyboard_listener 구현에서 설정한다고 가정
    else:
        piper = None
        wrist_rs_cam = exo_rs_cam = table_rs_cam = None
        event = {}

    # 3) LLP Policy 로딩
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id,
        cfg.train_dataset.root,
        revision=cfg.train_dataset.revision,
    )

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )

    policy, res = wrap_policy(
        policy=policy,
        cfg=cfg.method,
        is_master=True,
        device=device,
    )
    logging.info(res)
    policy.eval()

    # 4) 카메라 녹화 (옵션)
    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        table_rs_cam.start_recording()

    logging.info(
        colored(
            f"[LLP] Runtime initialized. use_devices={cfg.use_devices}, task={cfg.task}",
            "green",
            attrs=["bold"],
        )
    )

    return LLPRuntimeContext(
        cfg=cfg,
        device=device,
        policy=policy,
        piper=piper,
        table_rs_cam=table_rs_cam,
        wrist_rs_cam=wrist_rs_cam,
        exo_rs_cam=exo_rs_cam,
        keyboard_event=event,
    )


def llp_step(
    ctx: LLPRuntimeContext,
    task_text: str,
) -> Tuple[float, float]:
    """
    LLP 한 step:
      - 카메라/로봇 상태 읽어서 batch 생성
      - policy.select_action
      - ctrl_end_pose
    return:
      (t_action_pred, t_total)
    """
    t_start = time.time()

    batch = create_llp_batch(
        piper=ctx.piper,
        table_rs_cam=ctx.table_rs_cam,
        wrist_rs_cam=ctx.wrist_rs_cam,
        use_devices=ctx.cfg.use_devices,
        task=task_text,
        use_end_pose=True,
    )

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(ctx.device, non_blocking=True)

    t_pred_start = time.time()
    with torch.no_grad():
        action_pred = ctx.policy.select_action(batch).squeeze()
    t_pred_end = time.time()

    # end pose + gripper 포맷 (프로젝트 규칙에 맞춰 int 캐스팅)
    end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
    gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]

    if ctx.piper is not None:
        ctrl_end_pose(ctx.piper, end_pose_data, gripper_data)

    t_total = time.time() - t_start
    t_action_pred = t_pred_end - t_pred_start

    logging.info(
        colored(
            f"[LLP] step done | task={task_text} | t_pred={t_action_pred:.3f}s | t_total={t_total:.3f}s",
            "yellow",
            attrs=["bold"],
        )
    )

    return t_action_pred, t_total


def llp_send_zero(ctx: LLPRuntimeContext):
    """HLP status == DONE 일 때 zero posture로 보내는 헬퍼."""
    if ctx.cfg.use_devices and ctx.piper is not None:
        logging.info("[LLP] Sending zero configuration (HLP status == DONE).")
        set_zero_configuration(ctx.piper)
