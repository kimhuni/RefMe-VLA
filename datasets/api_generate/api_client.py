# api_client.py
from __future__ import annotations

import json
import os
import re
import base64
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

from datasets.api_generate.config import STATUS_SET, render_prompt


# =========================
# Data models
# =========================
@dataclass
class ApiRequest:
    """Provider-agnostic request payload."""
    task: str
    prev_desc: str
    prev_status: str
    images: List[Tuple[str, str]]  # [(role, path)], role in {"side", "wrist"}


@dataclass
class ApiResponse:
    """Normalized response for downstream use."""
    #desc_1: str
    #desc_2: str
    desc: str
    status_reasoning: str
    status: str  # DONE | NOT_DONE | UNCERTAIN


# =========================
# Client
# =========================
class ApiClient:
    """
    Simple, readable API client that supports Gemini / OpenAI providers.
    - Keeps prompt short (minified JSON spec) to reduce truncation.
    - One-shot call (no retry loops); relies on max_tokens and prompt brevity.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        prompt_id: str,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        self.prompt_id = prompt_id

    # Public entry: build prompt -> call provider -> parse structured output
    def call(self, req: ApiRequest) -> ApiResponse:
        prompt = render_prompt(self.prompt_id, req.task, req.prev_desc, req.prev_status)
        raw_text = self._invoke_provider(req.images, prompt)
        return self._parse_response(raw_text)

    # -------------------------
    # Provider dispatch
    # -------------------------
    def _invoke_provider(self, images: List[Tuple[str, str]], prompt: str) -> str:
        if self.provider == "gemini":
            return self._invoke_gemini(images, prompt)
        if self.provider == "openai":
            return self._invoke_openai(images, prompt)
        raise ValueError(f"Unknown provider: {self.provider}")

    # ===== Gemini =====
    def _invoke_gemini(self, images: List[Tuple[str, str]], prompt: str) -> str:
        """
        One-shot Gemini request.
        - First try with response_mime_type='application/json'
        - Fallback once without response_mime_type for older SDKs
        - Prints a concise line per frame in pipeline (pipeline handles printing desc/status).
        """
        import google.generativeai as genai  # type: ignore

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)

        # assemble parts: [{"text": prompt}, {"text":"[IMAGE:side]"}, {"inline_data":{...}}, {"text":"[IMAGE:wrist]"}, {"inline_data":{...}}]
        parts: List[object] = [{"text": prompt}]
        for role, path in self._sorted_images(images):
            parts.append({"text": f"[IMAGE:{role}]"})
            parts.append(self._gemini_image_part(path))

        model = genai.GenerativeModel(model_name=self.model)

        # First attempt: JSON mime
        try:
            resp = model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens,
                    response_mime_type="application/json",
                ),
            )
        except Exception:
            # Fallback attempt: no response_mime_type (older SDKs)
            resp = model.generate_content(
                parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens,
                ),
            )

        # Keep a minimal, readable debug print at provider level if needed:
        # (commented out per user preference to print only desc/status in pipeline)
        # print(f"[gemini] model={self.model} usage={getattr(resp, 'usage_metadata', None)}")

        text = self._extract_text_from_gemini(resp)
        return text

    @staticmethod
    def _gemini_image_part(path: str) -> Dict[str, Dict[str, bytes | str]]:
        mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        with open(path, "rb") as f:
            data = f.read()
        return {"inline_data": {"mime_type": mime, "data": data}}

    @staticmethod
    def _extract_text_from_gemini(resp) -> str:
        """
        Prefer resp.text; otherwise traverse candidate->content->parts for first .text.
        Raise if no text is available.
        """
        if hasattr(resp, "text") and resp.text:
            return resp.text

        candidates = getattr(resp, "candidates", None)
        if candidates:
            cand0 = candidates[0]
            content = getattr(cand0, "content", None)
            parts_out = getattr(content, "parts", []) if content else []
            for p in parts_out:
                if getattr(p, "text", None):
                    return p.text

        raise RuntimeError("Gemini returned no text content.")

    # ===== OpenAI =====
    def _invoke_openai(self, images: List[Tuple[str, str]], prompt: str) -> str:
        """
        One-shot OpenAI Chat Completions request with JSON mode (stable in latest SDK).
        """
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI(api_key=api_key)

        # Build multimodal content: text + images (data URLs)
        content: List[dict] = [{"type": "text", "text": prompt}]
        for role, path in self._sorted_images(images):
            mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "text", "text": f"[IMAGE:{role}]"})
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}})
            # content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}", "detail":"low"}})

        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        raw_text = resp.choices[0].message.content or ""
        print("raw text : ", repr(raw_text))
        print("\n================================================================================================")
        return raw_text

    # -------------------------
    # Parsing helpers
    # -------------------------
    def _parse_response(self, raw_text: str) -> ApiResponse:
        """
        Strict JSON first; if that fails, pull fields via regex.
        """
        # Normalize to the first JSON object if extra text is around it
        if not raw_text.strip().startswith("{"):
            lb = raw_text.find("{")
            rb = raw_text.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                raw_text = raw_text[lb:rb+1]

        # 1) Strict JSON
        try:
            obj = json.loads(raw_text)
            d1 = obj.get("desc", "").strip()
            d2 = obj.get("status_reasoning", "").strip()
            st = obj.get("status", "").strip().upper()

            if st not in STATUS_SET:
                raise ValueError("invalid status")
            return ApiResponse(d1, d2, st)
        except Exception:
            pass

        # 2) Fallback regex
        d1 = _re_search(raw_text, r'"?desc_1"?\s*[:=]\s*"([^"]+)"') or _re_search(raw_text, r'1\)\s*(.+)') or ""
        d2 = _re_search(raw_text, r'"?desc_2"?\s*[:=]\s*"([^"]+)"') or _re_search(raw_text, r'2\)\s*(.+)') or ""
        st = _re_search(raw_text, r'(DONE|NOT_DONE|PARTIALLY_DONE)') or "UNCERTAIN"

        return ApiResponse(d1.strip(), d2.strip(), st.strip())

    # -------------------------
    # Utilities
    # -------------------------
    @staticmethod
    def _sorted_images(images: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Sort by role to keep deterministic order: side -> wrist -> others."""
        order = {"side": 0, "wrist": 1}
        return sorted(images, key=lambda x: order.get(x[0], 99))

    @staticmethod
    def _b64_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


# ============ Regex helper ============
def _re_search(text: str, pat: str) -> str:
    m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""
