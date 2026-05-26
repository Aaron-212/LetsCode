from __future__ import annotations

import logging
from typing import Any

import httpx

from .constants import USER_AGENT
from .model.leetcode import LeetCodeQuestion, LeetCodeQuestionList


logger = logging.getLogger(__name__)


class LeetCodeFetcher:
    BASE_URL = "https://leetcode-api-pied.vercel.app"
    REQUEST_TIMEOUT = 15.0

    def __init__(self, base_url: str = BASE_URL) -> None:
        self.base_url = base_url

    async def get_all_problems(
        self, limit: int = 20, skip: int = 0
    ) -> LeetCodeQuestionList | None:
        response = await self._request("GET", "/problems")

        if response is None:
            return None

        problems = LeetCodeQuestionList.model_validate_json(response.text).root
        return LeetCodeQuestionList(problems[skip : skip + limit])

    async def get_problem(self, question_slug: int | str) -> LeetCodeQuestion | None:
        response = await self._request("GET", f"/problem/{question_slug}")

        if response is None:
            return None

        return LeetCodeQuestion.model_validate_json(response.text)

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response | None:
        headers = {"User-Agent": USER_AGENT, "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
            ) as client:
                response = await client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            logger.warning("LeetCode request failed: %s %s: %s", method, path, error)
            return None

        if response.status_code != httpx.codes.OK:
            logger.warning(
                "LeetCode request returned %s: %s %s",
                response.status_code,
                method,
                path,
            )
            return None

        return response
