"""
googleGenAIAPI.py
-----------------
Uses client.models.generate_content for ALL calls (text + image).
This is the correct approach per the official google-genai SDK docs —
chats.create() has limitations with multimodal content and history management.
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
        """
        Send a conversation to the Gemini API and return a response wrapper.
        Uses generate_content for ALL calls — handles both text and images.

        Accepts OpenAI-style messages:
          [{"role": "system"|"user"|"assistant", "content": str | list}, ...]

        Returns object with: response.choices[0].message["content"]
        """

        # ── 1. Build system instruction and contents list ──────────────────
        system_instruction = None
        contents = []   # list of types.Content

        for msg in messages:
            role    = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
                continue

            # Map OpenAI roles to Gemini roles
            gemini_role = "model" if role == "assistant" else "user"
            parts = self._build_parts(content)
            contents.append(types.Content(role=gemini_role, parts=parts))

        # ── 2. Build generation config ──────────────────────────────────────
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        # ── 3. Call API with retry + back-off ───────────────────────────────
        for attempt in range(self.retries):
            try:
                loop = asyncio.get_event_loop()

                # Capture in local vars for the closure
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
                error_str   = str(e)
                wait        = self._parse_retry_delay(error_str)

                if wait:
                    print(f"⏳ Rate limit — waiting {wait}s "
                          f"(attempt {attempt + 1}/{self.retries})...")
                    await asyncio.sleep(wait)
                elif attempt < self.retries - 1:
                    backoff = 2 ** attempt
                    print(f"Error (attempt {attempt + 1}): {e} "
                          f"— retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                else:
                    raise e

        raise RuntimeError("Max retries exceeded.")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_parts(self, content) -> list:
        """
        Convert OpenAI-style content into a list of Gemini Part objects.

        str   → [Part(text=...)]
        list  → each block converted:
                  {"type":"text",  "text":"..."}
                  {"type":"image", "source":{"type":"base64","media_type":"...","data":"..."}}
        """
        if isinstance(content, str):
            return [types.Part.from_text(text=content)]

        if isinstance(content, list):
            parts = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    parts.append(types.Part.from_text(text=block["text"]))
                elif btype == "image":
                    src = block["source"]
                    if src["type"] == "base64":
                        image_bytes = base64.b64decode(src["data"])
                        parts.append(
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=src["media_type"],
                            )
                        )
            return parts

        return [types.Part.from_text(text=str(content))]

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
