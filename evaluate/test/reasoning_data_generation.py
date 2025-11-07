#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VLM 라벨 생성 파이프라인 — Real-time Task Evaluation 데이터 자동 생성
- LeRobot 포맷의 영상(.mp4)에서 1FPS 프레임 추출 → API(Gemini/ChatGPT)로 k=3 응답 수집 → 다수결 → JSONL/CSV 저장
- 출력 형식: 두 문장(S1/S2) + STATUS(DONE/NOT DONE/UNCERTAIN), 실패 시 동일 프레임 1회 재질의(결정성↑)

요구사항
- Python 3.10+
- ffmpeg, pillow, pandas(optional)
- 구글: google-generativeai(또는 신규 Google GenAI SDK: `google-genai`), 오픈AI: openai SDK

환경변수
- GEMINI_API_KEY, OPENAI_API_KEY (필요시 둘 중 하나만)

참조
- OpenAI Responses/비전: https://platform.openai.com/docs/guides/images-vision
- OpenAI Python SDK: https://platform.openai.com/docs/libraries/python-sdk
- Gemini 이미지 입력: https://ai.google.dev/gemini-api/docs/image-understanding
- Gemini API Quickstart: https://ai.google.dev/gemini-api/docs/quickstart
"""

import argparse
import base64
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# 선택적: pandas가 있으면 CSV 병합/정리 편함
try:
    import pandas as pd  # noqa: F401
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# -----------------------------
# 프롬프트 템플릿 (JSON 출력 강제)
# -----------------------------
EVAL_PROMPT_JSON = (
    "You are a precise robotic manipulation evaluator.\n"
    "TASK: {task}\n"
    "SUBGOAL: {subgoal}\n"
    "PREV: {prev_desc}\n"
    "RULE: Two short sentences then a STATUS token.\n"
    "Return strict JSON with keys: s1, s2, status.\n"
    "Constraints: s1<=20 tokens, s2<=15 tokens. status in [DONE, NOT DONE, UNCERTAIN].\n"
    "Example: {\"s1\":\"the gripper touches the blue button\",\"s2\":\"subgoal satisfied\",\"status\":\"DONE\"}"
)

# -----------------------------
# 공통 유틸
# -----------------------------

def run_ffmpeg_extract(mp4_path: str, out_dir: Path, fps: float = 1.0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", mp4_path,
        "-vf", f"fps={fps}",
        str(out_dir / "frame_%05d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def list_frames(frames_dir: Path) -> List[Path]:
    return sorted(frames_dir.glob("frame_*.jpg"))


def b64_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def annotate_image(img_path: Path, caption: str, out_path: Path, margin=8):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    # 폰트 선택
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        base_font = ImageFont.truetype(font_path, 24)
    except Exception:
        base_font = ImageFont.load_default()

    # shrink-to-fit
    text = caption
    font_size = 28
    while font_size >= 12:
        font = ImageFont.truetype(getattr(base_font, 'path', "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), font_size) if hasattr(base_font, 'path') else base_font
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if tw <= W - 2 * margin and th <= int(0.45 * H):
            break
        font_size -= 2
    box_h = th + 2 * margin
    overlay = Image.new("RGBA", (W, box_h), (0, 0, 0, 140))
    img.paste(overlay, (0, H - box_h), overlay)
    draw = ImageDraw.Draw(img)
    draw.multiline_text((margin, H - box_h + margin), text, font=font, fill=(255, 255, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=95)


# -----------------------------
# 제공자 별 API 래퍼 (Gemini / OpenAI)
# -----------------------------
class GeminiClient:
    """Google Gemini API 래퍼 (이미지+텍스트 멀티모달)."""
    def __init__(self, model: str = "gemini-2.5-flash"):
        # 신 SDK(google-genai) 우선, 없을 경우 구버전(google.generativeai) 시도
        try:
            from google import genai  # type: ignore
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            self.variant = "genai"
        except Exception:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.client = genai
            self.variant = "generativeai"
        self.model = model

    def generate_k(self, image_b64: str, prompt: str, k: int, max_tokens: int = 80,
                   temperature: float = 0.6, top_p: float = 0.9) -> List[str]:
        results = []
        if self.variant == "genai":
            # 신 SDK
            img_part = {
                "mime_type": "image/jpeg",
                "data": base64.b64decode(image_b64),
            }
            for _ in range(k):
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[img_part, prompt],
                    config={
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_output_tokens": max_tokens,
                        "response_mime_type": "application/json",
                    },
                )
                results.append(resp.text)
        else:
            # 구 SDK (google.generativeai)
            model = self.client.GenerativeModel(self.model)
            img_part = self.client.upload_file(bytes_data=base64.b64decode(image_b64), mime_type="image/jpeg")
            for _ in range(k):
                resp = model.generate_content([
                    img_part,
                    prompt,
                ], generation_config={
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                })
                results.append(resp.text)
        return results


class OpenAIClient:
    """OpenAI Responses API 래퍼 (비전 지원 모델)."""
    def __init__(self, model: str = "gpt-4o-mini"):
        # 최신 SDK(OpenAI) 사용
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("`openai` 패키지를 설치하세요: pip install openai") from e
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate_k(self, image_b64: str, prompt: str, k: int, max_tokens: int = 80,
                   temperature: float = 0.6, top_p: float = 0.9) -> List[str]:
        # Responses API: vision 입력은 data:image/jpeg;base64,... 형태 또는 file id로 첨부 가능
        img_url = f"data:image/jpeg;base64,{image_b64}"
        results = []
        for _ in range(k):
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": img_url},
                        ],
                    }
                ],
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"},  # JSON 강제
            )
            # SDK 통일성 위해 text 항목 사용
            try:
                results.append(resp.output[0].content[0].text)
            except Exception:
                # 백업: 전체를 문자열화
                results.append(json.dumps(resp.to_dict()))
        return results


# -----------------------------
# 투표/파싱 로직
# -----------------------------

def safe_parse(json_text: str) -> Tuple[str, str, str]:
    """JSON 문자열에서 (s1, s2, status)를 파싱. 실패 시 UNCERTAIN."""
    try:
        obj = json.loads(json_text)
        s1 = str(obj.get("s1", "")).strip()
        s2 = str(obj.get("s2", "")).strip()
        status = str(obj.get("status", "UNCERTAIN")).upper().strip()
        if status not in {"DONE", "NOT DONE", "UNCERTAIN"}:
            status = "UNCERTAIN"
        return s1, s2, status
    except Exception:
        return "", "", "UNCERTAIN"


def majority_vote(cands: List[Tuple[str, str, str]]):
    # STATUS 다수결 → 동률 시 UNCERTAIN 우선순위 가장 낮음 (DONE/NOT DONE 우선)
    from collections import Counter
    status_counts = Counter([c[2] for c in cands])
    if not status_counts:
        return ("", "", "UNCERTAIN"), cands, {"DONE": 0, "NOT DONE": 0, "UNCERTAIN": 0}
    # 동률 처리: DONE > NOT DONE > UNCERTAIN 우선
    order = ["DONE", "NOT DONE", "UNCERTAIN"]
    max_count = max(status_counts.values())
    winners = [s for s, n in status_counts.items() if n == max_count]
    winners.sort(key=lambda s: order.index(s))
    target_status = winners[0]
    # 선택: target_status 중 가장 짧은 S1+S2
    filtered = [c for c in cands if c[2] == target_status]
    pick = min(filtered, key=lambda x: len(x[0]) + len(x[1]))
    return pick, cands, {k: status_counts.get(k, 0) for k in ["DONE", "NOT DONE", "UNCERTAIN"]}


# -----------------------------
# 메인 파이프라인
# -----------------------------

def process_video(
    mp4_path: Path,
    out_root: Path,
    provider: str,
    model_name: str,
    task: str,
    subgoal: str,
    prev_desc_init: str = "",
    fps: float = 1.0,
    k: int = 3,
    retry_on_uncertain: bool = True,
    annotate: bool = True,
):
    video_name = mp4_path.stem
    frames_dir = out_root / "frames" / video_name
    out_dir = out_root / "results" / video_name / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 프레임 추출
    if not frames_dir.exists() or not any(frames_dir.glob("frame_*.jpg")):
        run_ffmpeg_extract(str(mp4_path), frames_dir, fps=fps)
    frames = list_frames(frames_dir)

    # 2) 클라이언트 준비
    if provider == "gemini":
        client = GeminiClient(model=model_name)
    elif provider == "openai":
        client = OpenAIClient(model=model_name)
    else:
        raise ValueError("provider must be one of {gemini, openai}")

    # 3) 로그 파일
    jsonl_path = out_dir / f"{video_name}_{model_name}.jsonl"
    csv_path = out_dir / f"{video_name}_{model_name}.csv"
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "video", "frame", "task", "subgoal", "prev_desc", "s1", "s2", "status",
        "votes_DONE", "votes_NOT_DONE", "votes_UNCERTAIN", "latency_ms", "retry", "provider", "model", "image_path"
    ])
    csv_w.writeheader()

    prev_desc = prev_desc_init

    with open(jsonl_path, "w", encoding="utf-8") as jf:
        for idx, img_path in enumerate(frames):
            prompt = EVAL_PROMPT_JSON.format(task=task, subgoal=subgoal, prev_desc=prev_desc)
            img_b64 = b64_image(img_path)

            # k=3 수집
            t0 = time.time()
            raw_texts = client.generate_k(img_b64, prompt, k=k, max_tokens=80, temperature=0.6, top_p=0.9)
            lat_ms = int((time.time() - t0) * 1000)

            parsed = [safe_parse(t) for t in raw_texts]
            pick, cands, vote_map = majority_vote(parsed)
            s1, s2, status = pick

            # UNCERTAIN 1회 재질의(결정성↑)
            retried = False
            if status == "UNCERTAIN" and retry_on_uncertain:
                retried = True
                raw_texts2 = client.generate_k(img_b64, prompt, k=1, max_tokens=80, temperature=0.2, top_p=0.95)
                parsed2 = [safe_parse(t) for t in raw_texts2]
                pick2, _, vote_map2 = majority_vote(parsed2)
                # 재질의 결과가 UNCERTAIN이 아니면 교체
                if pick2[2] != "UNCERTAIN":
                    s1, s2, status = pick2
                    # 투표수 갱신(의미상 1표)
                    vote_map = {k: (vote_map.get(k, 0) + vote_map2.get(k, 0)) for k in ["DONE", "NOT DONE", "UNCERTAIN"]}

            # prev_desc 갱신: S1/S2만 유지(짧게)
            prev_desc = f"S1: {s1} | S2: {s2} | STATUS: {status}"

            # JSONL 기록
            rec = {
                "video": video_name,
                "frame": idx,
                "image_path": str(img_path),
                "task": task,
                "subgoal": subgoal,
                "prev_desc": prev_desc,
                "candidates": [
                    {"s1": a, "s2": b, "status": c} for (a, b, c) in cands
                ],
                "pick": {"s1": s1, "s2": s2, "status": status},
                "votes": vote_map,
                "latency_ms": lat_ms,
                "provider": provider,
                "model": model_name,
                "retry": retried,
            }
            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # CSV 요약 기록
            csv_w.writerow({
                "video": video_name,
                "frame": idx,
                "task": task,
                "subgoal": subgoal,
                "prev_desc": prev_desc,
                "s1": s1,
                "s2": s2,
                "status": status,
                "votes_DONE": vote_map.get("DONE", 0),
                "votes_NOT_DONE": vote_map.get("NOT DONE", 0),
                "votes_UNCERTAIN": vote_map.get("UNCERTAIN", 0),
                "latency_ms": lat_ms,
                "retry": retried,
                "provider": provider,
                "model": model_name,
                "image_path": str(img_path),
            })

            # 주석 이미지(optional)
            if annotate:
                out_img = out_dir / f"frame_{idx:05d}_{model_name}.jpg"
                annotate_image(img_path, f"S1: {s1}\nS2: {s2}\nSTATUS: {status}", out_img)

    csv_f.close()


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Real-time Task Evaluation 라벨 생성 파이프라인 (LeRobot→API)")
    p.add_argument("--video", type=str, required=True, help="LeRobot 비디오(.mp4) 경로")
    p.add_argument("--out_root", type=str, default="./result/VLM_gen", help="출력 루트 디렉토리")
    p.add_argument("--provider", type=str, choices=["gemini", "openai"], required=True)
    p.add_argument("--model", type=str, required=True, help="API 모델명 (예: gemini-2.5-flash, gpt-4o-mini 등)")
    p.add_argument("--task", type=str, required=True, help="예: Press the blue button on the table.")
    p.add_argument("--subgoal", type=str, required=True, help="plan_id=0에서 선택된 서브목표 1개")
    p.add_argument("--prev_desc", type=str, default="", help="초기 prev_desc (선택)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--no_annotate", action="store_true")
    args = p.parse_args()

    process_video(
        mp4_path=Path(args.video),
        out_root=Path(args.out_root),
        provider=args.provider,
        model_name=args.model,
        task=args.task,
        subgoal=args.subgoal,
        prev_desc_init=args.prev_desc,
        fps=args.fps,
        k=args.k,
        retry_on_uncertain=True,
        annotate=not args.no_annotate,
    )


if __name__ == "__main__":
    main()
