# evaluate/eval_real_time_API_LLP_pi0.py
from termcolor import colored

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging
import time
import torch

from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.utils.utils import (
    init_devices,
    get_safe_torch_device,
    init_keyboard_listener
)

from configs.default import DatasetConfig, EvalConfig, WandBConfig
from configs.policies import PreTrainedConfig
from common.utils.random_utils import set_seed
from common.robot_devices.robot_utils import read_end_pose_msg, ctrl_end_pose, read_joint_msg, set_zero_configuration
from common.policies.factory import make_policy, wrap_policy

from common.utils.utils import get_safe_torch_device
from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.policies.factory import make_policy, wrap_policy
from common.policies.pi0.modeling_pi0 import PI0Policy


# 필요하면 프로젝트에서 쓰는 config 타입으로 교체해도 됨
# (예: from common.configs.policy import PolicyConfig 등)
# 여기서는 Any로 두고 parser.wrap() 에서 채워준다고 가정
GRIPPER_EFFORT = 2500  # 프로젝트에 맞게 값 조정


# =========================
# 1. Config & Runtime 정의
# =========================

@dataclass
class LLPConfig:
    """
    pi0 LLP 쪽 설정.
    - use_devices: 실제 로봇/카메라 사용 여부
    - task: high-level task 문자열 ("press the blue button" 등)
    - max_steps: 최대 루프 스텝 수 (main에서 사용)
    - train_dataset / policy / method: 기존 eval_real_time_ours.py 에서 쓰던 설정 그대로
    """
    # 아래 세 개는 parser.wrap() / Hydra 가 채워주는 걸 가정
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
    infer_chunk: int = 45


    def __post_init__(self):
        if self.policy is not None:
            return

        if self.policy_path:
            # config.json을 읽어 적절한 서브클래스 인스턴스를 돌려줌 (type=pi0 등)
            self.policy = PreTrainedConfig.from_pretrained(self.policy_path)
            self.policy.pretrained_path = self.policy_path
            logging.info(f"[LLPConfig] Loaded policy cfg from: {self.policy_path} (type={self.policy.type})")
        else:
            logging.warning(
                "No policy_path provided; policy config will need to be set manually before make_policy()."
            )

@dataclass
class LLPRuntimeContext:
    """
    LLP(pi0)가 돌아가는 데 필요한 런타임 리소스를 한 군데에 모은 컨텍스트.
    main_realtime_hlp_llp 에서 import해서 그대로 사용.
    """
    cfg: LLPConfig
    device: torch.device

    policy: torch.nn.Module

    piper: Any
    table_rs_cam: Any
    wrist_rs_cam: Any
    # exo_rs_cam: Any

    keyboard_event: Dict[str, Any]


# =========================
# 2. batch 생성 유틸
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
    pi0 정책이 기대하는 입력 포맷에 맞게 batch 딕셔너리를 생성.
      - observation.state: EE pose 또는 joint
      - observation.images.table / wrist: 카메라 이미지
      - task: [task_str] (batch 차원 1개)
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
        # return {
        #     "observation.state": random_piper_action(),
        #     "observation.images.table": random_piper_image(),
        #     "observation.images.wrist": random_piper_image(),
        #     "task": [task],
        # }
        return {}


# =========================
# 3. pi0 Policy 초기화
# =========================

def init_llp_runtime(cfg: LLPConfig) -> LLPRuntimeContext:
    """
    pi0 LLP 정책 + 로봇/카메라/키보드 런타임 초기화.
    - main 쪽에서는 이 함수를 호출해서 LLPRuntimeContext만 받아서 사용.
    """

    # 1) 디바이스 & 시드
    device = get_safe_torch_device(cfg.device, log=True)
    torch.backends.cudnn.benchmark = True

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # 2) 로봇 & 카메라 & 키보드 이벤트 초기화
    if cfg.use_devices:
        piper, cam = init_devices(cfg)
        wrist_rs_cam = cam["wrist_rs_cam"]
        # exo_rs_cam = cam["exo_rs_cam"]
        table_rs_cam = cam["table_rs_cam"]

        listener, event, _task_state = init_keyboard_listener()
        # ESC → event["reset_hlp"] = True 같은 키 매핑은 init_keyboard_listener 안에서 처리한다고 가정
    else:
        piper = None
        wrist_rs_cam = exo_rs_cam = table_rs_cam = None
        event = {}

    # 3) 학습 데이터셋 메타 정보
    #    (eval_real_time_ours.py 에서 쓰던 형태 그대로 재사용)
    if cfg.train_dataset is None:
        raise ValueError("[LLP] cfg.train_dataset 가 설정되지 않았습니다.")

    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id,
        cfg.train_dataset.root,
        revision=getattr(cfg.train_dataset, "revision", None),
    )

    # 4) pi0 policy 생성
    if cfg.policy is None:
        raise ValueError("[LLP] cfg.policy 가 설정되지 않았습니다.")


    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )

    # 5) method(wrap) 적용 (LoRA / MSP / 등등)
    # if cfg.method is None:
    #     raise ValueError("[LLP] cfg.method 가 설정되지 않았습니다.")

    # policy, res = wrap_policy(
    #     policy=policy,
    #     cfg=cfg.method,
    #     is_master=True,
    #     device=device,
    # )
    # logging.info(res)
    policy.eval()

    # 6) 카메라 녹화 시작 (선택)
    if cfg.use_devices:
        wrist_rs_cam.start_recording()
        # exo_rs_cam.start_recording()
        table_rs_cam.start_recording()

    time.sleep(5)

    logging.info(
        colored(
            f"[LLP] Runtime initialized. "
            f"use_devices={cfg.use_devices}, task={cfg.task}, device={cfg.device}",
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
        # exo_rs_cam=exo_rs_cam,
        keyboard_event=event,
    )


# =========================
# 4. pi0 한 step 실행
# =========================

def llp_step(
    ctx: LLPRuntimeContext,
    task_text: str,
) -> Tuple[float, float]:
    """
    pi0 LLP 한 step:
      - 카메라/로봇 상태 읽어서 batch 생성
      - policy.select_action(batch)
      - ctrl_end_pose 로 실제 로봇 제어

    return:
      (t_action_pred, t_total)  # 추론 시간, 전체 step 시간
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

    # 텐서만 GPU로
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(ctx.device, non_blocking=True)

    t_pred_start = time.time()
    with torch.no_grad():
        action_pred = ctx.policy.select_action(batch).squeeze()
    # reset after "infer_chunk" action
    if len(ctx.policy._action_queue) < ctx.cfg.infer_chunk:
        ctx.policy.reset()

    t_pred_end = time.time()

    # end pose + gripper 포맷 (프로젝트 규칙에 맞게 int 캐스팅)
    end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
    gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]

    if ctx.piper is not None:
        ctrl_end_pose(ctx.piper, end_pose_data, gripper_data)

    t_total = time.time() - t_start
    t_action_pred = t_pred_end - t_pred_start

    logging.info(
        colored(
            f"[LLP] step done | task={task_text} | "
            f"t_pred={t_action_pred:.3f}s | t_total={t_total:.3f}s",
            "yellow",
            attrs=["bold"],
        )
    )

    return t_action_pred, t_total


# =========================
# 5. DONE 시 zero posture로
# =========================

def llp_send_zero(ctx: LLPRuntimeContext):
    """
    HLP status == DONE 이 되었을 때,
    로봇을 zero configuration 으로 보내는 함수.
    main 쪽에서 llp_task_input 이 None 일 때 한 번만 호출해주면 됨.
    """
    if ctx.cfg.use_devices and ctx.piper is not None:
        logging.info("[LLP] Sending zero configuration (HLP status == DONE).")
        set_zero_configuration(ctx.piper)