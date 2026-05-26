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


if __name__ == "__main__":
    unittest.main()
