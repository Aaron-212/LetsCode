import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

from letscode.leetcode import LeetCodeFetcher


class LeetCodeFetcherTest(unittest.TestCase):
    def test_get_problem_resolves_numeric_id_through_graphql(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.method, "POST")
            self.assertEqual(request.url.path, "/graphql")
            payload = json.loads(request.content)
            requests.append(payload)

            if payload["operationName"] == "problemsetQuestionList":
                self.assertEqual(
                    payload["variables"]["filters"],
                    {"searchKeywords": "1"},
                )
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "problemsetQuestionList": {
                                "total": 1,
                                "questions": [
                                    {
                                        "frontendQuestionId": "1",
                                        "title": "Two Sum",
                                        "titleSlug": "two-sum",
                                        "difficulty": "Easy",
                                        "paidOnly": False,
                                        "hasSolution": True,
                                    }
                                ],
                            }
                        }
                    },
                )

            return httpx.Response(
                200,
                json={
                    "data": {
                        "question": {
                            "questionId": "1",
                            "questionFrontendId": "1",
                            "title": "Two Sum",
                            "titleSlug": "two-sum",
                            "content": "<p>Return indices.</p>",
                            "difficulty": "Easy",
                            "hints": ["<p>Use a map.</p>"],
                            "isPaidOnly": False,
                        }
                    }
                },
            )

        fetcher = LeetCodeFetcher(transport=httpx.MockTransport(handler))

        question = asyncio.run(fetcher.get_problem(1))

        self.assertIsNotNone(question)
        self.assertEqual(question.questionId, 1)
        self.assertEqual(question.title, "Two Sum")
        self.assertEqual(question.hints, ["<p>Use a map.</p>"])
        self.assertEqual(
            [request["operationName"] for request in requests],
            ["problemsetQuestionList", "questionData"],
        )

    def test_get_problem_uses_slug_directly(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(payload["operationName"], "questionData")
            self.assertEqual(payload["variables"]["titleSlug"], "two-sum")
            return httpx.Response(
                200,
                json={
                    "data": {
                        "question": {
                            "questionId": "1",
                            "questionFrontendId": "1",
                            "title": "Two Sum",
                            "titleSlug": "two-sum",
                            "content": "<p>Return indices.</p>",
                            "difficulty": "Easy",
                            "hints": [],
                            "isPaidOnly": False,
                        }
                    }
                },
            )

        fetcher = LeetCodeFetcher(transport=httpx.MockTransport(handler))

        question = asyncio.run(fetcher.get_problem("two-sum"))

        self.assertIsNotNone(question)
        self.assertEqual(question.questionId, 1)
        self.assertEqual(len(requests), 1)

    def test_get_all_problems_maps_graphql_list_items(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            self.assertEqual(payload["operationName"], "problemsetQuestionList")
            self.assertEqual(payload["variables"]["limit"], 1)
            self.assertEqual(payload["variables"]["skip"], 5)
            return httpx.Response(
                200,
                json={
                    "data": {
                        "problemsetQuestionList": {
                            "total": 10,
                            "questions": [
                                {
                                    "frontendQuestionId": "2",
                                    "title": "Add Two Numbers",
                                    "titleSlug": "add-two-numbers",
                                    "difficulty": "Medium",
                                    "paidOnly": False,
                                    "hasSolution": True,
                                }
                            ],
                        }
                    }
                },
            )

        fetcher = LeetCodeFetcher(transport=httpx.MockTransport(handler))

        problems = asyncio.run(fetcher.get_all_problems(limit=1, skip=5))

        self.assertIsNotNone(problems)
        problem = problems.root[0]
        self.assertEqual(problem.questionId, 2)
        self.assertEqual(problem.title_slug, "add-two-numbers")
        self.assertEqual(
            problem.url,
            "https://leetcode.com/problems/add-two-numbers/",
        )

    def test_get_random_problem_id_uses_random_offset(self) -> None:
        requests: list[dict[str, object]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            requests.append(payload)

            skip = payload["variables"]["skip"]
            question = {
                "frontendQuestionId": "42",
                "title": "Trapping Rain Water",
                "titleSlug": "trapping-rain-water",
                "difficulty": "Hard",
                "paidOnly": False,
                "hasSolution": True,
            }
            return httpx.Response(
                200,
                json={
                    "data": {
                        "problemsetQuestionList": {
                            "total": 100,
                            "questions": [question] if skip == 41 else [],
                        }
                    }
                },
            )

        fetcher = LeetCodeFetcher(transport=httpx.MockTransport(handler))

        with patch("letscode.leetcode.random.randrange", return_value=41):
            question_id = asyncio.run(fetcher.get_random_problem_id())

        self.assertEqual(question_id, 42)
        self.assertEqual(
            [request["variables"]["skip"] for request in requests],
            [0, 41],
        )

    def test_get_random_problem_id_skips_paid_only_problem(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            skip = payload["variables"]["skip"]
            paid_only = skip == 3
            return httpx.Response(
                200,
                json={
                    "data": {
                        "problemsetQuestionList": {
                            "total": 100,
                            "questions": [
                                {
                                    "frontendQuestionId": "50" if paid_only else "51",
                                    "title": "Sample",
                                    "titleSlug": "sample",
                                    "difficulty": "Medium",
                                    "paidOnly": paid_only,
                                    "hasSolution": True,
                                }
                            ],
                        }
                    }
                },
            )

        fetcher = LeetCodeFetcher(transport=httpx.MockTransport(handler))

        with patch("letscode.leetcode.random.randrange", side_effect=[3, 4]):
            question_id = asyncio.run(fetcher.get_random_problem_id())

        self.assertEqual(question_id, 51)


if __name__ == "__main__":
    unittest.main()
