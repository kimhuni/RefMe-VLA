## evaluate/eval_real_time_LLP_pi0.py v3
from termcolor import colored

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging
import time
import torch

from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.utils.utils import init_devices, get_safe_torch_device
from configs.default import DatasetConfig, EvalConfig
from configs.policies import PreTrainedConfig
from common.utils.random_utils import set_seed
from common.robot_devices.robot_utils import (
    read_end_pose_msg, ctrl_end_pose, read_joint_msg, set_zero_configuration
)
from common.policies.factory import make_policy

GRIPPER_EFFORT = 500  # 프로젝트 값 유지


# =========================
# 1. Config & Runtime 정의
# =========================

@dataclass
class LLPConfig:
    train_dataset: DatasetConfig
    policy: Optional[PreTrainedConfig] = None
    eval: EvalConfig = field(default_factory=EvalConfig)

    policy_path: Optional[str] = None
    output_dir: Optional[str] = None
    use_devices: bool = True

    task: str = "press the blue button"
    max_steps: int = 1000
    seed: Optional[int] = None
    fps: int = 5
    cam_list: list[str] = field(default_factory=lambda: ['wrist', 'exo', 'table'])
    device: str = "cuda:0"
    infer_chunk: int = 40

    def __post_init__(self):
        if self.policy is not None:
            return
        if self.policy_path:
            self.policy = PreTrainedConfig.from_pretrained(self.policy_path)
            self.policy.pretrained_path = self.policy_path
            logging.info(f"[LLPConfig] Loaded policy cfg from: {self.policy_path} (type={self.policy.type})")
        else:
            logging.warning("No policy_path provided; policy config must be set before make_policy().")


@dataclass
class LLPRuntimeContext:
    cfg: LLPConfig
    device: torch.device
    policy: torch.nn.Module

    piper: Any
    table_rs_cam: Any
    wrist_rs_cam: Any
    exo_rs_cam: Any


# =========================
# 2. Batch & Shared Observation 유틸
# =========================

def create_llp_batch(
    piper,
    table_rs_cam,
    wrist_rs_cam,
    use_devices: bool,
    task: str,
    use_end_pose: bool = True,
) -> Dict[str, Any]:
    """
    (기존 호환용) LLP 내부에서 직접 센서/카메라 읽어서 batch 생성.
    """
    if use_devices:
        state = read_end_pose_msg(piper) if use_end_pose else read_joint_msg(piper)
        return {
            "observation.state": state,
            "observation.images.table": table_rs_cam.image_for_inference(),
            "observation.images.wrist": wrist_rs_cam.image_for_inference(),
            "task": [task],
        }
    else:
        return {}


def create_llp_batch_from_obs(
    state: torch.Tensor,
    table_img: torch.Tensor,
    wrist_img: torch.Tensor,
    task: str,
) -> Dict[str, Any]:
    """
    (신규) main에서 1회 캡처한 관측(state/images)으로 batch 패킹.
    HLP/LLP 동일 timestep 공유를 위해 사용.
    """
    return {
        "observation.state": state,
        "observation.images.table": table_img,
        "observation.images.wrist": wrist_img,
        "task": [task],
    }


def capture_shared_observation(
    piper,
    table_rs_cam,
    wrist_rs_cam,
    use_devices: bool,
    use_end_pose: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    main에서 HLP/LLP 공용으로 쓸 관측을 "한 번만" 캡처하는 헬퍼.
    """
    if not use_devices:
        return None, None, None

    state = read_end_pose_msg(piper) if use_end_pose else read_joint_msg(piper)
    table_img = table_rs_cam.image_for_inference()
    wrist_img = wrist_rs_cam.image_for_inference()
    return state, table_img, wrist_img


# =========================
# 3. pi0 Policy 초기화
# =========================

def init_llp_runtime(cfg: LLPConfig) -> LLPRuntimeContext:
    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 로봇/카메라 초기화
    if cfg.use_devices:
        piper, cam = init_devices(cfg)
        wrist_rs_cam = cam["wrist_rs_cam"]
        exo_rs_cam = cam["exo_rs_cam"]
        table_rs_cam = cam["table_rs_cam"]
    else:
        piper = None
        wrist_rs_cam = exo_rs_cam = table_rs_cam = None

    # 키보드는 main에서 task-group 매핑을 위해 관리 (여기서는 생성하지 않음)
    # event/listener는 main이 가진다.

    if cfg.train_dataset is None:
        raise ValueError("[LLP] cfg.train_dataset is not set")

    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id,
        cfg.train_dataset.root,
        revision=getattr(cfg.train_dataset, "revision", None),
    )

    llp_loading_start = time.time()

    if cfg.policy is None:
        raise ValueError("[LLP] cfg.policy is not set")

    policy = make_policy(cfg=cfg.policy, ds_meta=train_dataset_meta)
    policy.eval()

    print("[LLP] loading time: ", time.time() - llp_loading_start, "sec")

    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        exo_rs_cam.start_recording()
        table_rs_cam.start_recording()

    time.sleep(5)

    logging.info(
        colored(
            f"[LLP] Runtime initialized. use_devices={cfg.use_devices}, task={cfg.task}, device={cfg.device}",
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
    )


# =========================
# 4. pi0 한 step 실행
# =========================

def llp_step(
    ctx: LLPRuntimeContext,
    task_text: str,
    batch: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """
    (핵심) main이 batch를 만들어 넘기면 그걸 사용하고,
    batch=None이면 기존처럼 내부에서 create_llp_batch로 캡처한다.
    """
    t_start = time.time()

    if batch is None:
        batch = create_llp_batch(
            piper=ctx.piper,
            table_rs_cam=ctx.table_rs_cam,
            wrist_rs_cam=ctx.wrist_rs_cam,
            use_devices=ctx.cfg.use_devices,
            task=task_text,
            use_end_pose=True,
        )

    # 텐서만 GPU로
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(ctx.device, non_blocking=True)

    t_pred_start = time.time()
    with torch.no_grad():
        action_pred = ctx.policy.select_action(batch).squeeze()
    # t_pred_end = time.time()

    # # ✅ infer_chunk reset 로직은 절대 건드리지 않음 (요청사항)
    # if len(ctx.policy._action_queue) < ctx.cfg.infer_chunk:
    #     ctx.policy.reset()
    #
    # end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
    # gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]
    #
    # if ctx.piper is not None:
    #     ctrl_end_pose(ctx.piper, end_pose_data, gripper_data)

    # [modified] send action data until it reaches chunk_size
    counter = 0

    while not (len(ctx.policy._action_queue) < ctx.cfg.infer_chunk):
        counter += 1
        end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
        gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]

        if ctx.piper is not None:
            ctrl_end_pose(ctx.piper, end_pose_data, gripper_data)

        with torch.no_grad():
            action_pred = ctx.policy.select_action(batch).squeeze()

        # print(f"[LLP] [{counter}] action_pred: ", action_pred)

        time.sleep(0.2)

    ctx.policy.reset()

    t_pred_end = time.time()


    t_total = time.time() - t_start
    t_action_pred = t_pred_end - t_pred_start
    return t_action_pred, t_total


def llp_send_zero(ctx: LLPRuntimeContext):
    if ctx.cfg.use_devices and ctx.piper is not None:
        logging.info("[LLP] Sending zero configuration.")
        set_zero_configuration(ctx.piper)
        time.sleep(4)
        logging.info("[LLP] Done zero configuration.")