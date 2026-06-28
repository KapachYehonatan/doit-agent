import unittest
from unittest.mock import patch

from doit_agent.llm import ModelResponseError
from doit_agent.safety import (
    assess_command_safety_with_trace,
    find_known_modifying_command,
    is_known_read_only_command,
)
from doit_agent.types import LlmCall, ModelConfig, SafetyCompletion, SafetyDecision


class SafetyTests(unittest.TestCase):
    def test_deterministic_classification(self):
        modifying = {
            "rm old.txt": "rm",
            "find . -delete": "find -delete",
            "echo hi > file.txt": "redirection",
            "sed -i s/a/b/ file.txt": "sed -i",
        }
        for command, expected in modifying.items():
            with self.subTest(command=command):
                self.assertEqual(find_known_modifying_command(command), expected)
                self.assertFalse(is_known_read_only_command(command))

        read_only = [
            'find ~ -type d -name "testing_doit" 2>/dev/null',
            "find /tmp -maxdepth 1 -type f | wc -l",
            "ls -la | sort",
        ]
        for command in read_only:
            with self.subTest(command=command):
                self.assertIsNone(find_known_modifying_command(command))
                self.assertTrue(is_known_read_only_command(command))

        self.assertFalse(is_known_read_only_command("find . -exec ls {} ;"))

    def test_known_cases_do_not_call_llm(self):
        with patch("doit_agent.safety.judge_command_modifies_files_with_trace") as judge:
            dangerous = assess_command_safety_with_trace("touch note.txt")
            safe = assess_command_safety_with_trace("find /tmp -type f | wc -l")

        self.assertTrue(dangerous.decision.modifies_files)
        self.assertEqual(dangerous.source, "known_modifying:touch")
        self.assertFalse(safe.decision.modifies_files)
        self.assertEqual(safe.source, "known_read_only")
        judge.assert_not_called()

    def test_unknown_command_uses_llm_or_conservative_fallback(self):
        completion = SafetyCompletion(
            decision=SafetyDecision(False, "It only displays information."),
            llm_call=LlmCall(
                model=ModelConfig(name="ollama/gemma3:4b"),
                messages=[{"role": "user", "content": "custom-tool --inspect"}],
                raw_content='{"modifies_files":false,"explanation":"read only"}',
            ),
        )
        with patch(
            "doit_agent.safety.judge_command_modifies_files_with_trace",
            return_value=completion,
        ):
            assessment = assess_command_safety_with_trace("custom-tool --inspect")

        self.assertFalse(assessment.decision.modifies_files)
        self.assertEqual(assessment.source, "llm")
        self.assertEqual(assessment.llm_call, completion.llm_call)

        with patch(
            "doit_agent.safety.judge_command_modifies_files_with_trace",
            side_effect=ModelResponseError("bad json"),
        ):
            failed = assess_command_safety_with_trace("custom-tool --inspect")

        self.assertTrue(failed.decision.modifies_files)
        self.assertEqual(failed.source, "llm_failed_assume_dangerous")
        self.assertIn("confirmation is required", failed.decision.explanation)


if __name__ == "__main__":
    unittest.main()
