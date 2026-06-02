import tempfile
import unittest
from pathlib import Path

from letscode.storage import ProblemStore


class ProblemStoreStateTest(unittest.TestCase):
    def test_save_and_load_last_question_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProblemStore(Path(directory))

            store.save_last_question_id(42)

            self.assertEqual(store.load_last_question_id(), 42)

    def test_load_last_question_id_returns_none_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProblemStore(Path(directory))

            self.assertIsNone(store.load_last_question_id())

    def test_load_last_question_id_returns_none_for_invalid_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProblemStore(Path(directory))
            store.state_path.write_text("{", encoding="utf-8")

            self.assertIsNone(store.load_last_question_id())

    def test_save_and_load_generated_problem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProblemStore(Path(directory))

            store.save_generated_problem(-1, "# Custom Problem")
            stored_problem = store.load_problem(-1)

            self.assertIsNotNone(stored_problem)
            self.assertEqual(stored_problem.question_id, -1)
            self.assertEqual(stored_problem.problem, "# Custom Problem")
            self.assertEqual(stored_problem.fetched_hints, [])

    def test_next_custom_question_id_counts_down(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ProblemStore(Path(directory))

            self.assertEqual(store.next_custom_question_id(), -1)

            store.save_generated_problem(-1, "# First")
            store.save_generated_problem(-2, "# Second")
            store.save_fetched_problem(1, "# LeetCode", [])

            self.assertEqual(store.next_custom_question_id(), -3)

    def test_reset_all_removes_data_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "data"
            store = ProblemStore(data_path)
            store.save_last_question_id(42)
            store.save_generated_problem(-1, "# Custom Problem")
            judge_path = data_path / "judge" / "solutions" / "abc.py"
            judge_path.parent.mkdir(parents=True)
            judge_path.write_text("class Solution:\n    pass\n", encoding="utf-8")

            store.reset_all()

            self.assertFalse(data_path.exists())


if __name__ == "__main__":
    unittest.main()
