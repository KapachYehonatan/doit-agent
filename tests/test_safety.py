import unittest
from unittest.mock import patch

from doit_agent.safety import (
    assess_command_safety,
    assess_command_safety_with_trace,
    find_known_modifying_command,
)
from doit_agent.types import LlmCall, ModelConfig, SafetyCompletion, SafetyDecision


class SafetyTests(unittest.TestCase):
    def test_known_modifying_command_is_detected(self):
        self.assertEqual(find_known_modifying_command("rm old.txt"), "rm")
        self.assertEqual(find_known_modifying_command("sudo mv a b"), "mv")
        self.assertEqual(find_known_modifying_command("ls && mkdir out"), "mkdir")

    def test_read_only_command_is_not_known_modifying_command(self):
        self.assertIsNone(find_known_modifying_command("ls -la"))
        self.assertIsNone(find_known_modifying_command("grep needle file.txt"))

    def test_known_modifying_command_does_not_call_llm(self):
        with patch("doit_agent.safety.judge_command_modifies_files_with_trace") as judge:
            decision = assess_command_safety("touch note.txt")

        self.assertTrue(decision.modifies_files)
        self.assertIn("touch", decision.explanation)
        judge.assert_not_called()

    def test_unknown_command_asks_llm(self):
        completion = SafetyCompletion(
            decision=SafetyDecision(
                modifies_files=False,
                explanation="It only displays information.",
            ),
            llm_call=LlmCall(
                model=ModelConfig(name="ollama/gemma3:4b"),
                messages=[{"role": "user", "content": "git status"}],
                raw_content=(
                    '{"modifies_files":false,'
                    '"explanation":"It only displays information."}'
                ),
            ),
        )
        with patch(
            "doit_agent.safety.judge_command_modifies_files_with_trace",
            return_value=completion,
        ) as judge:
            assessment = assess_command_safety_with_trace("git status")

        self.assertFalse(assessment.decision.modifies_files)
        self.assertEqual(
            assessment.decision.explanation, "It only displays information."
        )
        self.assertEqual(assessment.source, "llm")
        self.assertEqual(assessment.llm_call, completion.llm_call)
        judge.assert_called_once_with("git status")


if __name__ == "__main__":
    unittest.main()
