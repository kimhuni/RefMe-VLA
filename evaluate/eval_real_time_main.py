# eval_real_time.py  (== main_realtime_hlp_llp.py)

from dataclasses import dataclass, field
from typing import Optional, Tuple

import logging
import time

from termcolor import colored

from lerobot.configs import parser
from common.utils.utils import init_logging

from evaluate.eval_real_time_HLP_qwen import HLPConfig, HighLevelPlanner
from evaluate.eval_real_time_LLP_pi0 import (
    LLPConfig,
    LLPRuntimeContext,
    init_llp_runtime,
    llp_step,
    llp_send_zero,
)


@dataclass
class EvalRealTimeMainConfig:
    """
    메인 오케스트레이터 config.
    - hlp: HLPConfig
    - llp: LLPConfig
    - use_hlp: HLP의 on/off 스위치
    """
    use_hlp: bool = True
    hlp: HLPConfig = field(default_factory=HLPConfig)
    llp: LLPConfig = field(default_factory=LLPConfig)


def decide_subtask_and_status(
    use_hlp: bool,
    hlp: Optional[HighLevelPlanner],
    global_task: str,
    side_img_tensor,
    wrist_img_tensor,
) -> Tuple[str, str]:
    """
    모든 모드는 이 함수만 통해서 (subtask_text, status)를 얻도록 통일.

    - use_hlp=True  → HLP를 실제로 돌려서 subtask/status 얻기
    - use_hlp=False → subtask = global_task, status="NOT_DONE" 고정
    """
    if use_hlp:
        hlp_out = hlp.step(
            task=global_task,
            side_img_tensor=side_img_tensor,
            wrist_img_tensor=wrist_img_tensor,
        )
        subtask = hlp_out.get("subtask") or global_task
        status = hlp_out.get("status", "UNCERTAIN")
        return subtask, status
    else:
        subtask = global_task
        status = "NOT_DONE"
        return subtask, status


@parser.wrap()
def eval_real_time_main(cfg: EvalRealTimeMainConfig):
    """
    HLP + LLP를 동시에 돌리는 메인 루프.
    `use_hlp` 플래그로 HLP on/off.
    """
    init_logging()
    logging.info(
        colored(
            f"[MAIN] Starting real-time loop | use_hlp={cfg.use_hlp} | task={cfg.llp.task}",
            "green",
            attrs=["bold"],
        )
    )

    # 1) LLP runtime 초기화 (로봇, 카메라, 정책, 키보드 등)
    llp_ctx: LLPRuntimeContext = init_llp_runtime(cfg.llp)

    # 2) HLP 초기화 (use_hlp=True일 때만)
    hlp: Optional[HighLevelPlanner] = None
    if cfg.use_hlp:
        hlp = HighLevelPlanner(cfg.hlp)

    step = 0
    done_sent_zero = False  # DONE 이후 zero posture 보냈는지 여부

    while True:
        t_loop_start = time.time()

        # 2-1) ESC 등으로 HLP 리셋 (키바인딩은 init_keyboard_listener에서 설정했다고 가정)
        event = llp_ctx.keyboard_event
        if cfg.use_hlp and event.get("reset_hlp", False):
            if hlp is not None:
                hlp.reset()
            done_sent_zero = False
            event["reset_hlp"] = False

        # 2-2) HLP/LLP 공통용 이미지 캡처
        if cfg.llp.use_devices:
            # HLP용: global view(SIDE)=exo, WRIST=wrist
            side_img_tensor = llp_ctx.exo_rs_cam.image_for_inference()
            wrist_img_tensor = llp_ctx.wrist_rs_cam.image_for_inference()
        # else:
            # 오프라인/디버그용: 랜덤 이미지 (필요시 별도 util로 변경)
            # from lerobot.scripts.eval_real_time_utils import random_piper_image
            #
            # side_img_tensor = random_piper_image()
            # wrist_img_tensor = random_piper_image()

        # 2-3) subtask / status 결정
        subtask_text, status = decide_subtask_and_status(
            use_hlp=cfg.use_hlp,
            hlp=hlp,
            global_task=cfg.llp.task,
            side_img_tensor=side_img_tensor,
            wrist_img_tensor=wrist_img_tensor,
        )

        # 2-4) status에 따라 LLP task 입력 결정
        if status == "DONE":
            llp_task_input = None
        else:
            llp_task_input = subtask_text

        # 2-5) DONE이면 zero posture로 보내고 LLP 동작 중단
        if llp_task_input is None:
            if not done_sent_zero:
                llp_send_zero(llp_ctx)
                done_sent_zero = True

            t_total = time.time() - t_loop_start
            logging.info(
                colored(
                    f"[MAIN][STEP {step}] status=DONE | subtask={subtask_text} | t_total={t_total:.3f}s",
                    "cyan",
                    attrs=["bold"],
                )
            )

            step += 1
            if step > cfg.llp.max_steps:
                break

            time.sleep(0.2)
            continue

        done_sent_zero = False  # 다시 동작 시작하므로 플래그 리셋

        # 2-6) LLP 한 step 실행
        t_pred, t_total_llp = llp_step(llp_ctx, task_text=llp_task_input)

        t_loop_total = time.time() - t_loop_start
        logging.info(
            colored(
                f"[MAIN][STEP {step}] status={status} | subtask={subtask_text} | "
                f"llp_task_input={llp_task_input} | t_pred={t_pred:.3f}s | "
                f"t_loop_total={t_loop_total:.3f}s",
                "magenta",
                attrs=["bold"],
            )
        )

        step += 1
        if step > cfg.llp.max_steps:
            break

        time.sleep(0.2)

    logging.info("[MAIN] Real-time loop finished.")
