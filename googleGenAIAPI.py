"""
googleGenAIAPI.py
-----------------
Uses client.models.generate_content for ALL calls (text + image).
"""

import asyncio
import base64
import re
import streamlit as st
from google import genai
from google.genai import types

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]


class GoogleGenAIAPI:
    def __init__(self, retries=5):
        self.retries = retries
        self.client = genai.Client(api_key=GOOGLE_API_KEY)

    async def chat_completion(self, model, messages, temperature, max_tokens):
        # ── 1. Parse messages ──────────────────────────────────────────────
        system_instruction = None
        contents = []

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = self._build_parts(content)
            if parts:
                contents.append(types.Content(role=gemini_role, parts=parts))

        # Gemini requires conversation to start with a user turn
        # and must alternate user/model. Fix if needed.
        contents = self._fix_turn_order(contents)

        # ── 2. Config ──────────────────────────────────────────────────────
        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        # ── 3. Call with retry ─────────────────────────────────────────────
        for attempt in range(self.retries):
            try:
                loop = asyncio.get_event_loop()
                _contents = contents
                _config   = config
                _model    = model

                def _call():
                    return self.client.models.generate_content(
                        model=_model,
                        contents=_contents,
                        config=_config,
                    )

                response = await loop.run_in_executor(None, _call)
                return _GeminiResponseWrapper(response.text)

            except Exception as e:
                # Log the FULL error so it shows in Streamlit Cloud logs
                print(f"[GoogleGenAI] Attempt {attempt+1}/{self.retries} "
                      f"| Model: {model} | Error: {type(e).__name__}: {e}")

                error_str = str(e)
                wait = self._parse_retry_delay(error_str)

                if wait:
                    print(f"[GoogleGenAI] Rate limit — waiting {wait}s...")
                    await asyncio.sleep(wait)
                elif attempt < self.retries - 1:
                    backoff = 2 ** attempt
                    await asyncio.sleep(backoff)
                else:
                    # Raise with full message visible to Streamlit
                    raise RuntimeError(
                        f"Gemini API error after {self.retries} attempts.\n"
                        f"Model: {model}\n"
                        f"Error: {type(e).__name__}: {e}"
                    ) from e

        raise RuntimeError("Max retries exceeded.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_parts(self, content) -> list:
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        if isinstance(content, list):
            parts = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(types.Part.from_text(text=text))
                elif btype == "image":
                    src = block.get("source", {})
                    if src.get("type") == "base64":
                        try:
                            image_bytes = base64.b64decode(src["data"])
                            parts.append(
                                types.Part.from_bytes(
                                    data=image_bytes,
                                    mime_type=src["media_type"],
                                )
                            )
                        except Exception as ex:
                            print(f"[GoogleGenAI] Image decode error: {ex}")
            return parts

        return [types.Part.from_text(text=str(content))]

    @staticmethod
    def _fix_turn_order(contents: list) -> list:
        """
        Gemini requires:
        1. First turn must be 'user'
        2. Turns must strictly alternate user/model
        This fixes any violations by merging or removing bad turns.
        """
        if not contents:
            return contents

        fixed = []
        for turn in contents:
            if not fixed:
                # First turn must be user
                if turn.role == "user":
                    fixed.append(turn)
                # Skip model turns at the start
            else:
                last_role = fixed[-1].role
                if turn.role != last_role:
                    # Correct alternation — add it
                    fixed.append(turn)
                else:
                    # Same role back-to-back — merge parts into last turn
                    merged_parts = fixed[-1].parts + turn.parts
                    fixed[-1] = types.Content(role=last_role, parts=merged_parts)

        # Must end with a user turn (last turn drives the response)
        while fixed and fixed[-1].role == "model":
            fixed.pop()

        return fixed

    @staticmethod
    def _parse_retry_delay(error_str: str):
        m = re.search(r'retry[^\d]*(\d+(?:\.\d+)?)s', error_str, re.IGNORECASE)
        return float(m.group(1)) + 2 if m else None


# ── Response wrapper ────────────────────────────────────────────────────────
class _GeminiResponseWrapper:
    def __init__(self, text: str):
        self.choices = [_Choice(text)]

class _Choice:
    def __init__(self, text: str):
        self.message = {"content": text}