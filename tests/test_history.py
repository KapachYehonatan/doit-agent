from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from doit_agent.history import (
    append_error_history,
    append_history,
    append_memories,
    append_report_history,
    current_session,
    format_history,
    format_memories,
    format_session_history,
    format_user_context,
    load_memories,
    load_recent_history,
    load_recent_shell_history,
    shell_history_path,
)


class HistoryTests(unittest.TestCase):
    def test_session_id_uses_env_or_cwd_fallback(self):
        with patch.dict("os.environ", {"DOIT_SESSION_ID": "window-1"}):
            self.assertEqual(current_session("/work"), {"id": "window-1", "source": "env"})
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                current_session("/work"),
                {"id": "cwd:/work", "source": "cwd"},
            )

    def test_jsonl_histories_append_cwd_session_and_skip_bad_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / ".doit" / "history.jsonl"
            report_path = Path(tmpdir) / ".doit" / "report_history.jsonl"
            error_path = Path(tmpdir) / ".doit" / "error_history.jsonl"

            with (
                patch("doit_agent.history.os.getcwd", return_value="/work"),
                patch.dict("os.environ", {"DOIT_SESSION_ID": "window-1"}),
            ):
                append_history({"instruction": "list", "kind": "command"}, path=history_path)
                append_report_history({"instruction": "report"}, path=report_path)
                append_error_history({"instruction": "bad"}, path=error_path)
            history_path.write_text(
                history_path.read_text(encoding="utf-8") + "not json\n",
                encoding="utf-8",
            )

            history = load_recent_history(path=history_path)
            report = load_recent_history(path=report_path)
            error = load_recent_history(path=error_path)

        self.assertEqual(history[0]["cwd"], "/work")
        self.assertEqual(history[0]["session_id"], "window-1")
        self.assertEqual(history[0]["instruction"], "list")
        self.assertEqual(report[0]["instruction"], "report")
        self.assertEqual(error[0]["instruction"], "bad")

    def test_memories_round_trip_and_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / ".doit" / "memories.jsonl"
            with (
                patch("doit_agent.history.os.getcwd", return_value="/work"),
                patch.dict("os.environ", {}, clear=True),
            ):
                append_memories(["project is ~/p"], "remember project", path=path)
            path.write_text(
                path.read_text(encoding="utf-8") + 'bad\n{"not_memory":"x"}\n',
                encoding="utf-8",
            )

            memories = load_memories(path=path)
            text = format_memories(memories)

        self.assertEqual(memories[0]["memory"], "project is ~/p")
        self.assertEqual(memories[0]["session_id"], "cwd:/work")
        self.assertEqual(memories[0]["source_instruction"], "remember project")
        self.assertIn("1. 'project is ~/p'", text)

    def test_shell_history_path_and_loading(self):
        with patch.dict("os.environ", {"HISTFILE": "/tmp/custom_history"}):
            self.assertEqual(shell_history_path(), Path("/tmp/custom_history"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history"
            path.write_text(
                ": 1711111111:0;cd project\n"
                "\n"
                'export GEMINI_API_KEY="secret"\n'
                "MY_TOKEN=abc123 doit list files\n"
                "doit previous\n"
                "python train.py\n",
                encoding="utf-8",
            )

            commands = load_recent_shell_history(limit=3, path=path)

        self.assertEqual(
            commands,
            ["cd project", "MY_TOKEN=[REDACTED] doit list files", "python train.py"],
        )

    def test_shell_history_truncates_and_skips_noisy_env_prefixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history"
            path.write_text(
                'PATH="$HOME/.local/bin:$PATH" OLLAMA_HOST=127.0.0.1:11434 '
                'OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 '
                'OLLAMA_MAX_QUEUE=1 OLLAMA_NO_CLOUD=1 EXTRA_ONE=xxxxxxxxxxxxxxxx '
                'EXTRA_TWO=xxxxxxxxxxxxxxxx EXTRA_THREE=xxxxxxxxxxxxxxxx '
                'nice -n 10 ollama serve\n'
                "echo " + ("x" * 250) + "\n",
                encoding="utf-8",
            )

            commands = load_recent_shell_history(path=path)

        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0].endswith("...[truncated]"))

    def test_context_formatters_keep_useful_output(self):
        user_context = format_user_context("/work", ["python train.py"])
        history = format_history(
            [
                {
                    "instruction": "list",
                    "kind": "command",
                    "command": "ls",
                    "stdout": "a.txt\n",
                    "stderr": "warn\n",
                }
            ]
        )

        self.assertIn("CURRENT DIRECTORY: /work", user_context)
        self.assertIn("1. 'python train.py'", user_context)
        self.assertIn("stdout: 'a.txt\\n'", history)
        self.assertIn("stderr: 'warn\\n'", history)

    def test_session_history_separates_current_and_other_sessions(self):
        text = format_session_history(
            [
                {
                    "session_id": "window-2",
                    "cwd": "/docs",
                    "instruction": "make year folders",
                    "kind": "command",
                    "command": "mkdir 2020",
                },
                {
                    "session_id": "window-1",
                    "cwd": "/code",
                    "instruction": "list files",
                    "kind": "command",
                    "command": "ls",
                    "stdout": "a.py\n",
                },
            ],
            "window-1",
            "/code",
        )

        assert text is not None
        self.assertIn("CURRENT DOIT SESSION HISTORY", text)
        self.assertIn("OTHER DOIT SESSIONS", text)
        self.assertLess(text.index("list files"), text.index("make year folders"))
        self.assertIn("session: 'window-2'", text)


if __name__ == "__main__":
    unittest.main()
