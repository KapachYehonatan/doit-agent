import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from doit_agent.cli import main
from doit_agent.llm import ModelResponseError
from doit_agent.types import (
    AgentCompletion,
    AgentResponse,
    LlmCall,
    ModelConfig,
    SafetyAssessment,
    SafetyDecision,
    ShellResult,
)


def planner_completion(response: AgentResponse, source: str = "json") -> AgentCompletion:
    return AgentCompletion(
        response=response,
        llm_call=LlmCall(
            model=ModelConfig(name="ollama/gemma3:4b"),
            messages=[{"role": "user", "content": "test"}],
            raw_content='{"kind":"command","command":"test"}',
        ),
        source=source,
    )


def safety_assessment(modifies: bool, explanation: str = "test") -> SafetyAssessment:
    return SafetyAssessment(
        decision=SafetyDecision(modifies_files=modifies, explanation=explanation),
        source="test",
    )


class CliTests(unittest.TestCase):
    def run_main(self, argv):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("doit_agent.history.Path.home", return_value=Path(tmpdir)),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_command_executes_and_records_output_history(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="printf hi")
                ),
            ),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(False, "It only prints."),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="hi", stderr="", returncode=0),
            ) as run_shell,
            patch("doit_agent.cli.append_history") as append_history,
        ):
            code, stdout, stderr = self.run_main(["say hi"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "printf hi\nhi")
        self.assertEqual(stderr, "")
        run_shell.assert_called_once_with("printf hi")
        self.assertEqual(append_history.call_args.args[0]["stdout"], "hi")

    def test_dangerous_command_needs_confirmation(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="rm old.txt")
                ),
            ),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(True, "It deletes files."),
            ),
            patch("builtins.input", return_value="n") as input_,
            patch("doit_agent.cli.run_shell") as run_shell,
        ):
            code, stdout, stderr = self.run_main(["delete old.txt"])

        self.assertEqual(code, 0)
        self.assertIn("It deletes files.", stdout)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        input_.assert_called_once_with("Proceed? [y/N] ")
        run_shell.assert_not_called()

    def test_answer_with_command_is_only_a_suggestion(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(
                        kind="answer",
                        message="Use long listing.",
                        command="ls -la",
                    )
                ),
            ),
            patch("doit_agent.cli.run_shell") as run_shell,
            patch("doit_agent.cli.assess_command_safety_with_trace") as safety,
        ):
            code, stdout, stderr = self.run_main(["how do I list files"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "Use long listing.\nls -la\n")
        self.assertEqual(stderr, "")
        run_shell.assert_not_called()
        safety.assert_not_called()

    def test_planner_error_records_debug_log(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                side_effect=ModelResponseError(
                    "Model did not return valid JSON",
                    raw_content='{"kind":"answer","message":"Based on',
                ),
            ),
            patch("doit_agent.cli.append_error_history") as append_error_history,
        ):
            code, stdout, stderr = self.run_main(["summarize"])

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Model did not return valid JSON", stderr)
        entry = append_error_history.call_args.args[0]
        self.assertEqual(entry["error"]["stage"], "planner_response")
        self.assertIn("raw_content", entry["error"])

    def test_context_and_memories_are_plumbed(self):
        with (
            patch("doit_agent.cli.os.getcwd", return_value="/work/project"),
            patch.dict("os.environ", {"DOIT_SESSION_ID": "window-1"}),
            patch(
                "doit_agent.cli.load_recent_history",
                return_value=[
                    {
                        "session_id": "window-1",
                        "cwd": "/work/project",
                        "instruction": "list files",
                        "kind": "command",
                    },
                    {
                        "session_id": "window-2",
                        "cwd": "/docs",
                        "instruction": "make year folders",
                        "kind": "command",
                    },
                ],
            ),
            patch("doit_agent.cli.load_recent_shell_history", return_value=["python train.py"]),
            patch(
                "doit_agent.cli.load_memories",
                return_value=[{"memory": "project is ~/p"}],
            ),
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(
                        kind="answer",
                        message="Remembered.",
                        memories=["prefers terse output"],
                    )
                ),
            ) as complete_instruction,
            patch("doit_agent.cli.append_memories") as append_memories,
            patch("doit_agent.cli.append_report_history") as append_report_history,
        ):
            code, stdout, stderr = self.run_main(["remember this"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "Remembered.\n")
        self.assertEqual(stderr, "")
        _, kwargs = complete_instruction.call_args
        self.assertIn("project is ~/p", kwargs["memory_context"])
        self.assertIn("CURRENT DIRECTORY: /work/project", kwargs["user_context"])
        self.assertIn("CURRENT DOIT SESSION HISTORY", kwargs["history_context"])
        self.assertIn("OTHER DOIT SESSIONS", kwargs["history_context"])
        append_memories.assert_called_once_with(
            ["prefers terse output"],
            "remember this",
        )
        report = append_report_history.call_args.args[0]
        self.assertEqual(report["session"], {"id": "window-1", "source": "env"})
        self.assertEqual(report["user_context"]["cwd"], "/work/project")

    def test_clarification_replans_and_executes(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["created", "modified"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                side_effect=[
                    planner_completion(clarify, source="tool_call"),
                    planner_completion(AgentResponse(kind="command", command="ls -t")),
                ],
            ) as complete_instruction,
            patch("builtins.input", return_value="2"),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(False, "It lists files."),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="files", stderr="", returncode=0),
            ) as run_shell,
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Sort by which date?", stdout)
        self.assertIn("ls -t\nfiles", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(complete_instruction.call_count, 2)
        self.assertIn("modified", complete_instruction.call_args.kwargs["clarification_context"])
        run_shell.assert_called_once_with("ls -t")

    def test_bad_clarification_answer_aborts(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["created", "modified"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(clarify),
            ),
            patch("builtins.input", return_value="9"),
            patch("doit_agent.cli.run_shell") as run_shell,
            patch("doit_agent.cli.assess_command_safety_with_trace") as safety,
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        run_shell.assert_not_called()
        safety.assert_not_called()


if __name__ == "__main__":
    unittest.main()
