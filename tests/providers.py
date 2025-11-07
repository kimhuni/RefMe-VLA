# providers.py
import base64
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict, Any

ProviderName = Literal["gemini", "openai"]

STATUS_SET = {"DONE", "NOT_DONE", "UNCERTAIN"}

@dataclass
class ApiRequest:
    task: str
    prev_desc: str
    images: List[Tuple[str, str]]  # [(role, path)], role in {"side","wrist"}

@dataclass
class ApiResponse:
    desc_1: str
    desc_2: str
    status: str

# -------------------------
# Prompt builder (공용)
# -------------------------
def build_user_prompt(task: str, prev_desc: str) -> str:
    return (
        f"TASK: {task}\n"
        f"PREV: {prev_desc}\n"
        "RULES:\n"
        "- Describe only visible evidence (contact, alignment, illumination).\n"
        "- Exactly TWO sentences, then one of {DONE, NOT_DONE, UNCERTAIN}.\n"
        "- No speculation beyond the frame.\n"
        "RETURN JSON: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"DONE|NOT_DONE|UNCERTAIN\"}"
    )

# -------------------------
# Gemini adapter
# -------------------------
def call_gemini(model: str, req: ApiRequest, temperature=0.2, top_p=0.9, max_output_tokens=128) -> ApiResponse:
    """
    Requires: pip install google-generativeai
    Env: GOOGLE_API_KEY
    Docs:
      - Passing multiple images & JSON 응답 유도 (이미지 멀티모달)  [oai_citation:0‡Google AI for Developers](https://ai.google.dev/gemini-api/docs/image-understanding?utm_source=chatgpt.com)
    """
    import google.generativeai as genai
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    genai.configure(api_key=api_key)

    # 이미지 바이트 로드
    def to_part(path: str) -> Dict[str, Any]:
        mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        with open(path, "rb") as f:
            data = f.read()
        return {"mime_type": mime, "data": data}

    parts: List[Any] = []
    # 역할 설명을 텍스트로 먼저 고정(멀티이미지 순서 명시)
    parts.append(build_user_prompt(req.task, req.prev_desc))
    # side, wrist 순서 고정
    role_order = {"side": 0, "wrist": 1}
    for role, path in sorted(req.images, key=lambda x: role_order.get(x[0], 99)):
        parts.append(f"[IMAGE:{role}]")  # 역할 힌트 (텍스트 파트)
        parts.append(to_part(path))      # 실제 이미지 파트

    model_obj = genai.GenerativeModel(model_name=model)
    # Some versions may not support response_mime_type in GenerationConfig.
    try:
        resp = model_obj.generate_content(
            [*parts],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
            ),
        )
        raw = resp.text if hasattr(resp, "text") else (resp.candidates[0].content.parts[0].text)
    except Exception:
        # Fallback without response_mime_type (older SDKs)
        resp = model_obj.generate_content(
            [*parts],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
            ),
        )
        raw = resp.text if hasattr(resp, "text") else (resp.candidates[0].content.parts[0].text)

    try:
        obj = json.loads(raw)
    except Exception:
        # 혹시 JSON 실패 시 안전 파싱
        obj = _fallback_to_json(raw)

    return _coerce_api_response(obj)

# -------------------------
# OpenAI adapter (Responses API)
# -------------------------
def call_openai(model: str, req: ApiRequest, temperature=0.2, top_p=0.9, max_output_tokens=128) -> ApiResponse:
    """
    Requires: pip install openai
    Env: OPENAI_API_KEY
    Docs:
      - 이미지 입력/비전 가이드, Responses API, JSON 모드/구조화 출력  [oai_citation:1‡OpenAI 플랫폼](https://platform.openai.com/docs/guides/images-vision?utm_source=chatgpt.com)
    """
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)

    def to_b64(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # side, wrist 순서 고정
    role_order = {"side": 0, "wrist": 1}
    sorted_imgs = sorted(req.images, key=lambda x: role_order.get(x[0], 99))

    input_items = [
        {"type": "input_text", "text": build_user_prompt(req.task, req.prev_desc)}
    ]
    for role, path in sorted_imgs:
        input_items.append({"type": "input_text", "text": f"[IMAGE:{role}]"})
        input_items.append({
            "type": "input_image",
            "image_data": {
                "data": to_b64(path),
                "mime_type": "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png",
            },
        })

    # JSON 강제
    response = client.responses.create(
        model=model,
        input=input_items,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        response_format={"type": "json_object"},
    )
    # Responses API 표준 필드
    raw = response.output_text
    try:
        obj = json.loads(raw)
    except Exception:
        obj = _fallback_to_json(raw)

    return _coerce_api_response(obj)

# -------------------------
# helpers
# -------------------------
def _fallback_to_json(text: str) -> Dict[str, str]:
    # 아주 단순한 유사-JSON 파싱
    import re
    def grab(pat, default=""):
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        return (m.group(1).strip() if m else default)
    d1 = grab(r'"?desc_1"?\s*[:=]\s*"([^"]+)"')
    d2 = grab(r'"?desc_2"?\s*[:=]\s*"([^"]+)"')
    st = grab(r'(DONE|NOT_DONE|UNCERTAIN)', "UNCERTAIN").upper()
    return {"desc_1": d1, "desc_2": d2, "status": st}

def _coerce_api_response(obj: Dict[str, str]) -> ApiResponse:
    d1 = (obj.get("desc_1") or "").strip()
    d2 = (obj.get("desc_2") or "").strip()
    st = (obj.get("status") or "").strip().upper()
    if st not in STATUS_SET:
        st = "UNCERTAIN"
    return ApiResponse(d1, d2, st)

# -------------------------
# public entry
# -------------------------
def call_provider(provider: ProviderName, model: str, req: ApiRequest, **kw) -> ApiResponse:
    if provider == "gemini":
        return call_gemini(model, req, **kw)
    if provider == "openai":
        return call_openai(model, req, **kw)
    raise ValueError(f"Unknown provider: {provider}")