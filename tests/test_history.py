from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from doit_agent.history import (
    append_history,
    append_report_history,
    format_history,
    load_recent_history,
    report_history_path,
)


class HistoryTests(unittest.TestCase):
    def test_missing_history_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"

            self.assertEqual(load_recent_history(path=path), [])

    def test_append_history_creates_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".doit" / "history.jsonl"
            with patch("doit_agent.history.os.getcwd", return_value="/work"):
                append_history(
                    {
                        "instruction": "list files",
                        "kind": "command",
                        "command": "ls",
                        "executed": True,
                        "returncode": 0,
                    },
                    path=path,
                )

            entries = load_recent_history(path=path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["instruction"], "list files")
        self.assertEqual(entries[0]["cwd"], "/work")
        self.assertEqual(entries[0]["command"], "ls")

    def test_append_report_history_creates_separate_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".doit" / "report_history.jsonl"
            with patch("doit_agent.history.os.getcwd", return_value="/work"):
                append_report_history(
                    {
                        "agent_version": "part4_multi_turn",
                        "instruction": "list files",
                        "planner": {
                            "llm_call": {
                                "model": {
                                    "name": "ollama/gemma3:4b",
                                    "api_base": "http://localhost:11434",
                                },
                                "user_input": "list files",
                                "raw_content": '{"kind":"command","command":"ls"}',
                            }
                        },
                    },
                    path=path,
                )

            entries = load_recent_history(path=path)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["agent_version"], "part4_multi_turn")
        self.assertEqual(entries[0]["cwd"], "/work")
        self.assertEqual(
            entries[0]["planner"]["llm_call"]["model"]["name"],
            "ollama/gemma3:4b",
        )
        self.assertEqual(
            entries[0]["planner"]["llm_call"]["user_input"],
            "list files",
        )
        self.assertNotIn("messages", entries[0]["planner"]["llm_call"])

    def test_report_history_path_uses_report_filename(self):
        with patch("doit_agent.history.Path.home", return_value=Path("/home/user")):
            self.assertEqual(
                report_history_path(),
                Path("/home/user/.doit/report_history.jsonl"),
            )

    def test_load_recent_history_limits_and_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.jsonl"
            path.write_text(
                '{"instruction":"one"}\n'
                'not json\n'
                '{"instruction":"two"}\n'
                '{"instruction":"three"}\n',
                encoding="utf-8",
            )

            entries = load_recent_history(limit=2, path=path)

        self.assertEqual([entry["instruction"] for entry in entries], ["two", "three"])

    def test_format_history_summarizes_entries(self):
        text = format_history(
            [
                {
                    "instruction": "list files",
                    "kind": "command",
                    "command": "ls",
                    "executed": True,
                    "returncode": 0,
                },
                {
                    "instruction": "what can you do",
                    "kind": "answer",
                    "message": "I can run commands.",
                },
            ]
        )

        self.assertIn("user: 'list files'", text)
        self.assertIn("command: 'ls'", text)
        self.assertIn("message: 'I can run commands.'", text)


if __name__ == "__main__":
    unittest.main()
