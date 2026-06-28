from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from doit_agent.llm import (
    CLARIFICATION_TOOL,
    ConfigurationError,
    DEFAULT_MODEL,
    ModelResponseError,
    _complete_json,
    _model_config,
    _planner_messages,
    complete_instruction_with_trace,
    parse_clarification_tool_call,
    parse_model_content,
    parse_safety_model_content,
)
from doit_agent.types import ModelConfig


class ParseTests(unittest.TestCase):
    def test_model_content_accepts_main_response_shapes(self):
        command = parse_model_content(
            '{"kind":"command","command":"ls","memories":["project is ~/p"]}'
        )
        self.assertEqual(command.command, "ls")
        self.assertEqual(command.memories, ["project is ~/p"])

        answer = parse_model_content(
            '{"kind":"answer","message":"Use ls.","command":"ls -la"}'
        )
        self.assertEqual(answer.message, "Use ls.")
        self.assertEqual(answer.command, "ls -la")

        clarify = parse_model_content(
            '{"kind":"clarify","question":"Which file?",'
            '"options":["a.txt","b.txt"]}'
        )
        self.assertEqual(clarify.kind, "clarify")
        self.assertEqual(clarify.options, ["a.txt", "b.txt"])

    def test_model_content_extracts_first_json_object(self):
        response = parse_model_content(
            'Here is JSON:\n{"kind":"command","command":"ls"}\n```junk```'
        )

        self.assertEqual(response.kind, "command")
        self.assertEqual(response.command, "ls")

    def test_invalid_model_content_raises(self):
        for content in ["nope", '{"kind":"answer","message":""}', '{"kind":"wat"}']:
            with self.subTest(content=content):
                with self.assertRaises(ModelResponseError):
                    parse_model_content(content)

    def test_safety_content_accepts_and_rejects_expected_shapes(self):
        safe = parse_safety_model_content(
            'preface\n{"modifies_files":false,"explanation":"It reads."}'
        )
        self.assertFalse(safe.modifies_files)

        with self.assertRaises(ModelResponseError):
            parse_safety_model_content('{"modifies_files":"no"}')

    def test_clarification_tool_call(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "ask_clarification",
                                    "arguments": (
                                        '{"question":"Which date?",'
                                        '"options":["created","modified"]}'
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }

        parsed = parse_clarification_tool_call(response)

        assert parsed is not None
        self.assertEqual(parsed.question, "Which date?")
        self.assertEqual(parsed.options, ["created", "modified"])


class PromptAndConfigTests(unittest.TestCase):
    def test_planner_messages_include_contexts_before_instruction(self):
        messages = _planner_messages(
            "sort them",
            history_context="1. user: 'list files'",
            memory_context="1. 'project is ~/p'",
            user_context="CURRENT DIRECTORY: /tmp",
        )

        self.assertEqual(messages[-1], {"role": "user", "content": "sort them"})
        joined = "\n".join(message["content"] for message in messages[:-1])
        self.assertIn("SESSION-AWARE DOIT HISTORY", joined)
        self.assertIn("current-session history by default", joined)
        self.assertIn("USER MEMORIES", joined)
        self.assertIn("USER SHELL CONTEXT", joined)

    def test_model_config_precedence_and_api_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "[model]\n"
                "name = ollama/qwen3:4b-instruct\n"
                "api_base = http://localhost:11434\n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {"DOIT_MODEL": "ollama/gemma3:4b"}, clear=True),
            ):
                self.assertEqual(
                    _model_config(),
                    ModelConfig(
                        name="ollama/qwen3:4b-instruct",
                        api_base="http://localhost:11434",
                    ),
                )
                self.assertEqual(
                    _model_config("ollama/gemma3:4b"),
                    ModelConfig(name="ollama/gemma3:4b"),
                )

    def test_model_config_fallback_and_bad_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {"GEMINI_API_KEY": "secret"}, clear=True),
            ):
                self.assertEqual(_model_config(), ModelConfig(name=DEFAULT_MODEL))

            Path(tmpdir, "doit.cfg").write_text("[model]\nname = \n", encoding="utf-8")
            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {}, clear=True),
            ):
                with self.assertRaises(ConfigurationError):
                    _model_config()


class CompletionTests(unittest.TestCase):
    def test_complete_json_retries_supported_fallbacks_only(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("response_format is not supported")
            if "tools" in kwargs:
                raise RuntimeError("tools are not supported")
            return {"ok": True}

        fake_litellm = types.SimpleNamespace(completion=fake_completion)
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            result = _complete_json(
                config=ModelConfig(
                    name="ollama/qwen3:4b-instruct",
                    api_base="http://localhost:11434",
                ),
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=25,
                tools=[CLARIFICATION_TOOL],
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls[0]["api_base"], "http://localhost:11434")
        self.assertIn("response_format", calls[0])
        self.assertIn("tools", calls[1])
        self.assertNotIn("response_format", calls[1])
        self.assertNotIn("tools", calls[2])

    def test_complete_instruction_retries_incomplete_json_once(self):
        with (
            patch(
                "doit_agent.llm._complete_json",
                side_effect=[
                    {"choices": [{"message": {"content": '{"kind":"answer"'}}]},
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": '{"kind":"answer","message":"Done."}'
                                }
                            }
                        ]
                    },
                ],
            ) as complete_json,
            patch(
                "doit_agent.llm._model_config",
                return_value=ModelConfig(name="ollama/gemma3:4b"),
            ),
        ):
            completion = complete_instruction_with_trace("summarize")

        self.assertEqual(completion.response.message, "Done.")
        self.assertEqual(complete_json.call_count, 2)
        self.assertEqual(completion.llm_call.raw_content, '{"kind":"answer","message":"Done."}')


if __name__ == "__main__":
    unittest.main()
