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


def safety_assessment(decision: SafetyDecision) -> SafetyAssessment:
    return SafetyAssessment(decision=decision, source="test")


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

    def test_command_response_executes_shell(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="printf hi")
                ),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="hi", stderr="", returncode=0),
            ) as run_shell,
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(
                    SafetyDecision(
                        modifies_files=False,
                        explanation="It only prints text.",
                    )
                ),
            ),
        ):
            code, stdout, stderr = self.run_main(["say hi"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "printf hi\nhi")
        self.assertEqual(stderr, "")
        run_shell.assert_called_once_with("printf hi")

    def test_modifying_command_requires_confirmation(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="rm old.txt")
                ),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="", stderr="", returncode=0),
            ) as run_shell,
            patch("builtins.input", return_value="n") as input_,
        ):
            code, stdout, stderr = self.run_main(["delete old.txt"])

        self.assertEqual(code, 0)
        self.assertIn("rm old.txt\n", stdout)
        self.assertIn("can delete files or directories", stdout)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        input_.assert_called_once_with("Proceed? [y/N] ")
        run_shell.assert_not_called()

    def test_modifying_command_executes_after_confirmation(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="mkdir notes")
                ),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="", stderr="", returncode=0),
            ) as run_shell,
            patch("builtins.input", return_value="y"),
        ):
            code, stdout, stderr = self.run_main(["make a notes folder"])

        self.assertEqual(code, 0)
        self.assertIn("mkdir notes\n", stdout)
        self.assertIn("can create directories", stdout)
        self.assertEqual(stderr, "")
        run_shell.assert_called_once_with("mkdir notes")

    def test_answer_response_does_not_execute_shell(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="answer", message="I can run commands.")
                ),
            ),
            patch("doit_agent.cli.run_shell") as run_shell,
        ):
            code, stdout, stderr = self.run_main(["what can you do"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "I can run commands.\n")
        self.assertEqual(stderr, "")
        run_shell.assert_not_called()

    def test_cannot_do_response_does_not_execute_shell(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="cannot_do", message="I cannot do that.")
                ),
            ),
            patch("doit_agent.cli.run_shell") as run_shell,
        ):
            code, stdout, stderr = self.run_main(["change the moon"])

        self.assertEqual(code, 0)
        self.assertEqual(stdout, "I cannot do that.\n")
        self.assertEqual(stderr, "")
        run_shell.assert_not_called()

    def test_invalid_json_reports_error(self):
        with patch(
            "doit_agent.cli.complete_instruction_with_trace",
            side_effect=ModelResponseError("Model did not return valid JSON: nope"),
        ):
            code, stdout, stderr = self.run_main(["list files"])

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Model did not return valid JSON", stderr)

    def test_shell_failure_preserves_stderr_and_returncode(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="bad-command")
                ),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="", stderr="not found\n", returncode=127),
            ),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(
                    SafetyDecision(
                        modifies_files=False,
                        explanation="It only tries to run a command.",
                    )
                ),
            ),
        ):
            code, stdout, stderr = self.run_main(["run bad command"])

        self.assertEqual(code, 127)
        self.assertEqual(stdout, "bad-command\n")
        self.assertEqual(stderr, "not found\n")

    def test_command_response_records_history(self):
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="printf hi")
                ),
            ),
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="hi", stderr="", returncode=0),
            ),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(
                    SafetyDecision(
                        modifies_files=False,
                        explanation="It only prints text.",
                    )
                ),
            ),
            patch("doit_agent.cli.append_history") as append_history,
        ):
            self.run_main(["say hi"])

        append_history.assert_called_once_with(
            {
                "instruction": "say hi",
                "kind": "command",
                "command": "printf hi",
                "executed": True,
                "returncode": 0,
            }
        )

    def test_request_sends_available_history(self):
        with (
            patch(
                "doit_agent.cli.load_recent_history",
                return_value=[
                    {
                        "instruction": "delete file.txt",
                        "kind": "command",
                        "command": "rm file.txt",
                    }
                ],
            ),
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(
                    AgentResponse(kind="command", command="touch file.txt")
                ),
            ) as complete_instruction,
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="", stderr="", returncode=0),
            ),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(
                    SafetyDecision(
                        modifies_files=False,
                        explanation="It only creates an empty file.",
                    )
                ),
            ),
        ):
            self.run_main(["create a new empty .txt file"])

        _, kwargs = complete_instruction.call_args
        self.assertIn("delete file.txt", kwargs["history_context"])

    def test_tool_call_clarification_with_numeric_answer_replans_and_executes(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["creation date", "access date"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                side_effect=[
                    planner_completion(clarify, source="tool_call"),
                    planner_completion(
                        AgentResponse(kind="command", command="printf sorted")
                    ),
                ],
            ) as complete_instruction,
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="sorted", stderr="", returncode=0),
            ) as run_shell,
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(
                    SafetyDecision(
                        modifies_files=False,
                        explanation="It only prints text.",
                    )
                ),
            ),
            patch("builtins.input", return_value="2"),
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Sort by which date?", stdout)
        self.assertIn("1. creation date", stdout)
        self.assertIn("2. access date", stdout)
        self.assertIn("printf sorted\nsorted", stdout)
        self.assertEqual(stderr, "")
        run_shell.assert_called_once_with("printf sorted")
        self.assertEqual(complete_instruction.call_count, 2)
        _, kwargs = complete_instruction.call_args
        self.assertIn("access date", kwargs["clarification_context"])

    def test_json_clarification_with_free_text_answer_replans(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["creation date", "access date"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                side_effect=[
                    planner_completion(clarify),
                    planner_completion(AgentResponse(kind="command", command="ls -t")),
                ],
            ) as complete_instruction,
            patch(
                "doit_agent.cli.run_shell",
                return_value=ShellResult(stdout="", stderr="", returncode=0),
            ),
            patch(
                "doit_agent.cli.assess_command_safety_with_trace",
                return_value=safety_assessment(
                    SafetyDecision(
                        modifies_files=False,
                        explanation="It only lists files.",
                    )
                ),
            ),
            patch("builtins.input", return_value="modification date"),
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("ls -t\n", stdout)
        self.assertEqual(stderr, "")
        _, kwargs = complete_instruction.call_args
        self.assertIn("modification date", kwargs["clarification_context"])

    def test_empty_clarification_answer_aborts_without_execution(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["creation date", "access date"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(clarify),
            ),
            patch("doit_agent.cli.run_shell") as run_shell,
            patch("doit_agent.cli.assess_command_safety_with_trace") as safety,
            patch("builtins.input", return_value=""),
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        run_shell.assert_not_called()
        safety.assert_not_called()

    def test_out_of_range_clarification_answer_aborts(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["creation date", "access date"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(clarify),
            ),
            patch("doit_agent.cli.run_shell") as run_shell,
            patch("doit_agent.cli.assess_command_safety_with_trace") as safety,
            patch("builtins.input", return_value="9"),
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        run_shell.assert_not_called()
        safety.assert_not_called()

    def test_repeated_clarification_aborts_without_asking_again(self):
        clarify = AgentResponse(
            kind="clarify",
            question="Sort by which date?",
            options=["creation date", "access date"],
        )
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                return_value=planner_completion(clarify),
            ) as complete_instruction,
            patch("doit_agent.cli.run_shell") as run_shell,
            patch("doit_agent.cli.assess_command_safety_with_trace") as safety,
            patch("builtins.input", return_value="1") as input_,
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(complete_instruction.call_count, 2)
        input_.assert_called_once_with("> ")
        run_shell.assert_not_called()
        safety.assert_not_called()

    def test_clarification_loop_stops_after_three_distinct_questions(self):
        clarifications = [
            AgentResponse(
                kind="clarify",
                question="Sort by which date?",
                options=["creation date", "access date"],
            ),
            AgentResponse(
                kind="clarify",
                question="Ascending or descending?",
                options=["ascending", "descending"],
            ),
            AgentResponse(
                kind="clarify",
                question="Include hidden files?",
                options=["yes", "no"],
            ),
        ]
        with (
            patch(
                "doit_agent.cli.complete_instruction_with_trace",
                side_effect=[
                    planner_completion(response) for response in clarifications
                ],
            ) as complete_instruction,
            patch("doit_agent.cli.run_shell") as run_shell,
            patch("doit_agent.cli.assess_command_safety_with_trace") as safety,
            patch("builtins.input", side_effect=["1", "1", "1"]),
        ):
            code, stdout, stderr = self.run_main(["sort files"])

        self.assertEqual(code, 0)
        self.assertIn("Aborted.", stdout)
        self.assertEqual(stderr, "")
        self.assertEqual(complete_instruction.call_count, 3)
        run_shell.assert_not_called()
        safety.assert_not_called()


if __name__ == "__main__":
    unittest.main()
