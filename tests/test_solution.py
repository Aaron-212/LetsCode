import tempfile
import unittest
from pathlib import Path

from letscode.solution import (
    editor_command,
    ensure_solution_file,
    read_solution_file,
    write_solution_file,
)


class SolutionFileTest(unittest.TestCase):
    def test_ensure_solution_file_creates_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "solution.py"

            ensure_solution_file(path, "initial\n")
            ensure_solution_file(path, "replacement\n")

            self.assertEqual(read_solution_file(path), "initial\n")

    def test_write_solution_file_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "solution.py"

            write_solution_file(path, "class Solution:\n    pass\n")

            self.assertEqual(read_solution_file(path), "class Solution:\n    pass\n")

    def test_editor_command_splits_configured_editor(self) -> None:
        path = Path("/tmp/solution.py")

        self.assertEqual(
            editor_command(path, "code --wait"),
            ["code", "--wait", str(path)],
        )

    def test_editor_command_requires_editor(self) -> None:
        with self.assertRaises(RuntimeError):
            editor_command(Path("/tmp/solution.py"), "")


if __name__ == "__main__":
    unittest.main()
