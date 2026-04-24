"""
EduBot.py — Synchronous Bot class for Streamlit.
"""

import asyncio
from googleGenAIAPI import GoogleGenAIAPI
from prompts import (
    bot_sys_prompt, bot_prompt,
    translator_sys_prompt, translator_prompt,
    format_content_sys_prompt, format_content_prompt,
)

# Use one reliable model for everything
TEXT_MODEL   = "gemini-2.5-flash"
VISION_MODEL = "gemini-2.5-flash"


class Bot:
    def __init__(self):
        self.llm = GoogleGenAIAPI()

    # ── Public sync API ────────────────────────────────────────────────────

    def format_content_sync(self, content: str) -> str:
        return asyncio.run(self._format_content(content))

    def translate_sync(self, content: str, lang: str) -> str:
        return asyncio.run(self._translate(content, lang))

    def get_bot_response_sync(self, messages: list) -> str:
        return asyncio.run(self._get_response(messages))

    def build_initial_messages(self, problem: str, solution: str) -> list:
        return [
            {"role": "system", "content": bot_sys_prompt},
            {"role": "user",   "content": bot_prompt(problem=problem, solution=solution)},
        ]

    # ── Internal async ─────────────────────────────────────────────────────

    async def _get_response(self, messages: list) -> str:
        # Pick vision model if any message has an image
        has_image = any(
            isinstance(m.get("content"), list) and
            any(b.get("type") == "image" for b in m["content"])
            for m in messages
        )
        model = VISION_MODEL if has_image else TEXT_MODEL

        completion = await self.llm.chat_completion(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=1024,
        )
        return completion.choices[0].message["content"]

    async def _translate(self, content: str, lang: str) -> str:
        res = await self.llm.chat_completion(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": translator_sys_prompt},
                {"role": "user",   "content": translator_prompt(content=content, lang=lang)},
            ],
            temperature=0,
            max_tokens=512,
        )
        return res.choices[0].message["content"]

    async def _format_content(self, content: str) -> str:
        res = await self.llm.chat_completion(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": format_content_sys_prompt},
                {"role": "user",   "content": format_content_prompt(content=content)},
            ],
            temperature=0,
            max_tokens=512,
        )
        return res.choices[0].message["content"]