# eval_real_time_API_main.py  (== main_realtime_hlp_llp.py)

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

import io
import base64
import logging
import time

import numpy as np
from PIL import Image
import requests

from termcolor import colored

from configs.default import DatasetConfig
from common.utils.utils import init_logging

from evaluate.eval_real_time_API.eval_real_time_API_HLP_qwen import HLPConfig, HighLevelPlanner
from evaluate.eval_real_time_API.eval_real_time_API_LLP_pi0 import (
    LLPConfig,
    LLPRuntimeContext,
    init_llp_runtime,
    llp_step,
    llp_send_zero,
)


# ---------------------------
# Dataclass for main config
# ---------------------------

@dataclass
class EvalRealTimeMainConfig:
    """
    메인 오케스트레이터 config.
    - use_hlp: HLP 사용 여부
    - use_remote_hlp: FastAPI 서버로 원격 호출할지 여부
    - hlp_url: 원격 HLP 서버 주소 (use_remote_hlp=True일 때 사용)
    - hlp_period: HLP 호출 주기 (N 스텝마다 1회 호출)
    - hlp: HLP 설정 (로컬 HLP 사용할 때)
    - llp: LLP 설정
    """
    use_hlp: bool = True
    use_remote_hlp: bool = True
    hlp_url: str = "http://127.0.0.1:8787"
    hlp_period: int = 5

    hlp: HLPConfig = field(default_factory=HLPConfig)
    llp: LLPConfig = field(default_factory=LLPConfig)


# ---------------------------
# Utilities
# ---------------------------

def _pil_from_any(x) -> Image.Image:
    try:
        import torch
        _has_torch = True
    except Exception:
        _has_torch = False

    if isinstance(x, Image.Image):
        return x.convert("RGB")

    # torch.Tensor -> numpy
    if _has_torch and isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        # (1,C,H,W) -> (C,H,W)
        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        # (C,H,W) -> (H,W,C)
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        x = t.numpy()

    if isinstance(x, np.ndarray):
        arr = x
        # (1,H,W,C) -> (H,W,C)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        # (C,H,W) -> (H,W,C)
        if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        # float -> uint8
        if arr.dtype != np.uint8:
            amin, amax = float(arr.min()), float(arr.max())
            if amax <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        # 흑백 -> 3채널
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return Image.fromarray(arr, mode="RGB")

    raise TypeError(f"Unsupported image type: {type(x)}")



def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call_hlp_remote(
    url: str,
    task: str,
    prev_desc: str,
    prev_status: str,
    side_pil: Image.Image,
    wrist_pil: Image.Image,
    timeout: float = 15.0,
) -> Dict[str, Any]:
    """FastAPI HLP 서버로 /infer 요청"""
    payload = {
        "task": task,
        "prev_desc": prev_desc,
        "prev_status": prev_status,
        "side_b64": _pil_to_b64(side_pil),
        "wrist_b64": _pil_to_b64(wrist_pil),
    }
    resp = requests.post(url.rstrip("/") + "/infer", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()  # {desc_1, desc_2, status, subtask, raw_text, latency_ms}


# ---------------------------
# Main loop
# ---------------------------

def eval_real_time_main(cfg: EvalRealTimeMainConfig):
    """
    HLP + LLP를 동시에 돌리는 메인 루프.
    - use_hlp=True & use_remote_hlp=True  → FastAPI 서버 호출
    - use_hlp=True & use_remote_hlp=False → 로컬 HighLevelPlanner 사용
    - use_hlp=False → HLP 미사용, LLP는 global task를 반복 수행
    """
    init_logging()
    logging.info(
        colored(
            f"[MAIN] Starting real-time loop | use_hlp={cfg.use_hlp} | "
            f"use_remote_hlp={cfg.use_remote_hlp} | task={cfg.llp.task}",
            "green",
            attrs=["bold"],
        )
    )

    # 1) LLP runtime 초기화 (로봇, 카메라, 정책, 키보드 등)
    llp_ctx: LLPRuntimeContext = init_llp_runtime(cfg.llp)

    # 2) HLP 초기화
    hlp_local: Optional[HighLevelPlanner] = None
    if cfg.use_hlp and not cfg.use_remote_hlp:
        hlp_local = HighLevelPlanner(cfg.hlp)

    # HLP 누적 상태 (텍스트 히스토리)
    prev_desc, prev_status = "", "NOT_DONE"

    step = 0
    sent_zero_after_done = False  # DONE 이후 zero posture 보냈는지

    # 주 루프
    while True:
        t_loop_start = time.time()

        # ESC/RESET 키 처리 (키 이름 케이스/스페이스 안전 처리)
        event = llp_ctx.keyboard_event
        reset_flag = False
        for k in list(event.keys()):
            kk = str(k).lower().replace(" ", "")
            if event.get(k) and kk in ("esc", "reset", "setinitial"):
                reset_flag = True
                event[k] = False
        if reset_flag:
            prev_desc, prev_status = "", "NOT_DONE"
            sent_zero_after_done = False
            try:
                if cfg.use_hlp and cfg.use_remote_hlp:
                    requests.post(cfg.hlp_url.rstrip("/") + "/reset", timeout=2.0)
            except Exception as e:
                logging.warning(f"[MAIN] HLP /reset failed (ignored): {e}")

        # 2-1) 이미지 캡처
        if cfg.llp.use_devices:
            side_img_tensor = llp_ctx.table_rs_cam.image_for_inference()
            wrist_img_tensor = llp_ctx.wrist_rs_cam.image_for_inference()
        else:
            # 디바이스 미사용 모드라면, 여기서 오프라인 이미지 공급 로직을 넣으세요.
            logging.error("[MAIN] use_devices=False 모드는 아직 구현되지 않았습니다.")
            break

        # 2-2) HLP 호출 (주기적으로)
        status: str = prev_status
        subtask_text: str = cfg.llp.task  # 기본값(로컬/비HLP 모드)
        if cfg.use_hlp:
            if step % max(1, cfg.hlp_period) == 0:
                if cfg.use_remote_hlp:
                    print("sending hlp request")
                    # 원격 호출
                    side_pil = _pil_from_any(side_img_tensor)
                    wrist_pil = _pil_from_any(wrist_img_tensor)
                    try:
                        out = call_hlp_remote(
                            url=cfg.hlp_url,
                            task=cfg.llp.task,
                            prev_desc=prev_desc,
                            prev_status=prev_status,
                            side_pil=side_pil,
                            wrist_pil=wrist_pil,
                        )
                        print("raw_output: ", out)
                        status = out.get("status", "UNCERTAIN")
                        subtask_text = out.get("subtask") or cfg.llp.task
                        desc1 = out.get("desc_1", "")
                        desc2 = out.get("desc_2", "")
                        latency = out.get("latency_ms", None)
                        logging.info(
                            colored(
                                f"[HLP][STEP {step}] status={status} | subtask={subtask_text} | "
                                f"latency={latency} ms | desc1={desc1} | desc2={desc2}",
                                "yellow",
                            )
                        )
                        prev_desc, prev_status = f"{desc1} {desc2}", status
                    except Exception as e:
                        logging.error(f"[HLP] remote call failed: {e}")
                        # 실패 시 이전 상태 유지
                        status = prev_status
                        subtask_text = cfg.llp.task
                else:
                    # 로컬 HLP
                    out = hlp_local.step(
                        task=cfg.llp.task,
                        side_img_tensor=side_img_tensor,
                        wrist_img_tensor=wrist_img_tensor,
                    )
                    status = out.get("status", "UNCERTAIN")
                    subtask_text = out.get("subtask") or cfg.llp.task
                    desc1 = out.get("desc_1", "")
                    desc2 = out.get("desc_2", "")
                    logging.info(
                        colored(
                            f"[HLP][STEP {step}] status={status} | subtask={subtask_text} | "
                            f"desc1={desc1} | desc2={desc2}",
                            "yellow",
                        )
                    )
                    prev_desc, prev_status = f"{desc1} {desc2}", status
            else:
                # HLP 미호출 스텝에서는 직전 값 유지 (status/subtask)
                subtask_text = subtask_text if 'subtask_text' in locals() else cfg.llp.task
        else:
            # HLP 미사용 모드
            status = "NOT_DONE"
            subtask_text = cfg.llp.task

        # 2-3) status에 따라 LLP task 입력 결정
        if status == "DONE":
            llp_task_input = None
        else:
            llp_task_input = subtask_text

        # 2-4) DONE이면 zero posture 전송 (1회)
        if llp_task_input is None:
            if not sent_zero_after_done:
                llp_send_zero(llp_ctx)
                sent_zero_after_done = True

            t_total = time.time() - t_loop_start
            logging.info(
                colored(
                    f"[MAIN][STEP {step}] status=DONE | subtask={subtask_text} | t_total={t_total:.3f}s",
                    "cyan",
                    attrs=["bold"],
                )
            )
        else:
            sent_zero_after_done = False
            # 2-5) LLP 한 step 실행
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

        time.sleep(0.1)  # 주기 살짝 조정

    logging.info("[MAIN] Real-time loop finished.")


# ---------------------------
# CLI Entrypoint
# ---------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    # HLP on/off & remote
    p.add_argument("--use_hlp", type=int, default=1)
    p.add_argument("--use_remote_hlp", type=int, default=1)
    p.add_argument("--hlp_url", type=str, default="http://127.0.0.1:8787")
    p.add_argument("--hlp_period", type=int, default=5)

    # LLP high-level settings
    p.add_argument("--task", type=str, default="press the blue button")
    p.add_argument("--max_steps", type=int, default=1000)
    p.add_argument("--use_devices", type=int, default=1)
    p.add_argument("--llp_device", type=str, default="cuda:0")

    # LLP policy/dataset paths
    p.add_argument("--llp_model_path", type=str, required=True)
    p.add_argument("--dataset_repo_id", type=str, default=None)
    p.add_argument("--dataset_root", type=str, default=None)

    # (선택) 로컬 HLP에 필요한 경로들 (remote 사용 시 무시)
    p.add_argument("--hlp_model_path", type=str, default=None)
    p.add_argument("--hlp_adapter_path", type=str, default=None)
    p.add_argument("--hlp_use_qlora", type=int, default=1)
    p.add_argument("--hlp_device", type=str, default="cuda:0")

    args = p.parse_args()

    # HLP config (로컬 모드에서만 사용)
    hlp_cfg = HLPConfig(
        base_model_path=args.hlp_model_path,
        adapter_path=args.hlp_adapter_path,
        is_qlora=bool(args.hlp_use_qlora),
        device=args.hlp_device,
    )

    # LLP config
    llp_cfg = LLPConfig(
        train_dataset=DatasetConfig(
            repo_id=args.dataset_repo_id,
            root=args.dataset_root,
        ),
        policy_path=args.llp_model_path,
        use_devices=bool(args.use_devices),
        task=args.task,
        max_steps=args.max_steps,
        device=args.llp_device,
    )

    cfg = EvalRealTimeMainConfig(
        use_hlp=bool(args.use_hlp),
        use_remote_hlp=bool(args.use_remote_hlp),
        hlp_url=args.hlp_url,
        hlp_period=int(args.hlp_period),
        hlp=hlp_cfg,
        llp=llp_cfg,
    )

    eval_real_time_main(cfg)