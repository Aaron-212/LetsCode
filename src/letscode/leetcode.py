from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

from .constants import USER_AGENT
from .model.leetcode import LeetCodeQuestion, LeetCodeQuestionList

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProblemsetQuestionPage:
    total: int
    questions: list[dict[str, Any]]


class LeetCodeFetcher:
    BASE_URL = "https://leetcode.com"
    GRAPHQL_PATH = "/graphql"
    REQUEST_TIMEOUT = 15.0
    QUESTION_DATA_QUERY = """\
query questionData($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    questionId
    questionFrontendId
    title
    titleSlug
    content
    difficulty
    hints
    isPaidOnly
  }
}
"""
    QUESTION_LIST_QUERY = """\
query problemsetQuestionList(
  $categorySlug: String
  $limit: Int
  $skip: Int
  $filters: QuestionListFilterInput
) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      frontendQuestionId: questionFrontendId
      title
      titleSlug
      difficulty
      paidOnly: isPaidOnly
      hasSolution
    }
  }
}
"""
    LEGACY_QUESTION_LIST_QUERY = """\
query problemsetQuestionList(
  $categorySlug: String
  $limit: Int
  $skip: Int
  $filters: QuestionListFilterInput
) {
  problemsetQuestionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total
    questions {
      frontendQuestionId
      title
      titleSlug
      difficulty
      paidOnly
      hasSolution
    }
  }
}
"""

    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self.transport = transport

    async def get_all_problems(
        self, limit: int = 20, skip: int = 0
    ) -> LeetCodeQuestionList | None:
        questions = await self._get_problemset_questions(limit=limit, skip=skip)
        if questions is None:
            return None

        return LeetCodeQuestionList(
            [self._normalize_question_list_item(question) for question in questions]
        )

    async def get_random_problem_id(self, max_attempts: int = 20) -> int | None:
        page = await self._get_problemset_question_page(limit=1, skip=0)
        if page is None or page.total < 1:
            return None

        for _ in range(max_attempts):
            page = await self._get_problemset_question_page(
                limit=1,
                skip=random.randrange(page.total),
            )
            if page is None:
                return None

            if not page.questions:
                continue

            question = page.questions[0]
            if bool(question.get("paidOnly")):
                continue

            question_id = self._normalize_question_id(
                question.get("frontendQuestionId")
            )
            if question_id > 0:
                return question_id

        return None

    async def get_problem(self, question_slug: int | str) -> LeetCodeQuestion | None:
        title_slug = await self._resolve_title_slug(question_slug)
        if title_slug is None:
            return None

        data = await self._graphql(
            query=self.QUESTION_DATA_QUERY,
            variables={"titleSlug": title_slug},
            operation_name="questionData",
        )
        question = data.get("question") if data is not None else None
        if not isinstance(question, dict):
            return None

        try:
            return LeetCodeQuestion.model_validate(self._normalize_question(question))
        except ValidationError as error:
            logger.warning("LeetCode GraphQL question payload is invalid: %s", error)
            return None

    async def _resolve_title_slug(self, question_slug: int | str) -> str | None:
        if isinstance(question_slug, str) and not question_slug.isdecimal():
            return question_slug

        question_id = int(question_slug)
        questions = await self._get_problemset_questions(
            limit=50,
            skip=0,
            filters={"searchKeywords": str(question_id)},
        )
        if questions is None:
            return None

        for question in questions:
            if (
                self._normalize_question_id(question.get("frontendQuestionId"))
                == question_id
            ):
                title_slug = question.get("titleSlug")
                if isinstance(title_slug, str) and title_slug:
                    return title_slug

        return None

    async def _get_problemset_questions(
        self,
        *,
        limit: int,
        skip: int,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        page = await self._get_problemset_question_page(
            limit=limit,
            skip=skip,
            filters=filters,
        )
        return page.questions if page is not None else None

    async def _get_problemset_question_page(
        self,
        *,
        limit: int,
        skip: int,
        filters: dict[str, Any] | None = None,
    ) -> ProblemsetQuestionPage | None:
        variables = {
            "categorySlug": "",
            "limit": limit,
            "skip": skip,
            "filters": filters or {},
        }
        data = await self._graphql(
            query=self.QUESTION_LIST_QUERY,
            variables=variables,
            operation_name="problemsetQuestionList",
        )
        if data is None:
            data = await self._graphql(
                query=self.LEGACY_QUESTION_LIST_QUERY,
                variables=variables,
                operation_name="problemsetQuestionList",
            )

        problemset = data.get("problemsetQuestionList") if data is not None else None
        questions = (
            problemset.get("questions") if isinstance(problemset, dict) else None
        )
        total = problemset.get("total") if isinstance(problemset, dict) else None
        if not isinstance(questions, list) or not isinstance(total, int):
            return None

        return ProblemsetQuestionPage(
            total=total,
            questions=[
                question for question in questions if isinstance(question, dict)
            ],
        )

    async def _graphql(
        self,
        *,
        query: str,
        variables: dict[str, Any],
        operation_name: str,
    ) -> dict[str, Any] | None:
        response = await self._request(
            "POST",
            self.GRAPHQL_PATH,
            json={
                "query": query,
                "variables": variables,
                "operationName": operation_name,
            },
        )
        if response is None:
            return None

        try:
            payload = response.json()
        except ValueError:
            logger.warning("LeetCode GraphQL returned invalid JSON")
            return None

        if "errors" in payload:
            logger.warning("LeetCode GraphQL returned errors: %s", payload["errors"])
            return None

        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        return data

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response | None:
        headers = {
            "User-Agent": USER_AGENT,
            "Content-Type": "application/json",
            "Origin": self.BASE_URL,
            "Referer": f"{self.BASE_URL}/problemset/",
        }

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=self.REQUEST_TIMEOUT,
                transport=self.transport,
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

    def _normalize_question(self, question: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(question)
        question_id = self._normalize_question_id(question.get("questionFrontendId"))
        if question_id == 0:
            question_id = self._normalize_question_id(question.get("questionId"))

        normalized["questionId"] = question_id
        normalized["hints"] = question.get("hints") or []
        return normalized

    def _normalize_question_list_item(
        self,
        question: dict[str, Any],
    ) -> dict[str, Any]:
        title_slug = str(question.get("titleSlug", ""))
        return {
            "frontend_id": self._normalize_question_id(
                question.get("frontendQuestionId")
            ),
            "title": str(question.get("title", "")),
            "title_slug": title_slug,
            "difficulty": str(question.get("difficulty", "")),
            "url": f"https://leetcode.com/problems/{title_slug}/",
            "paid_only": bool(question.get("paidOnly")),
            "has_solution": bool(question.get("hasSolution")),
        }

    def _normalize_question_id(self, value: Any) -> int:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return 0
