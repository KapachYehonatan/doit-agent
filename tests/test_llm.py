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
    complete_instruction_with_trace,
    judge_command_modifies_files_with_trace,
    _model_config,
    _planner_messages,
    parse_clarification_tool_call,
    parse_model_content,
    parse_safety_model_content,
)
from doit_agent.types import ModelConfig


class ParseModelContentTests(unittest.TestCase):
    def test_parse_command(self):
        response = parse_model_content('{"kind":"command","command":"ls"}')
        self.assertEqual(response.kind, "command")
        self.assertEqual(response.command, "ls")

    def test_parse_answer_from_fenced_json(self):
        response = parse_model_content(
            '```json\n{"kind":"answer","message":"Hello"}\n```'
        )
        self.assertEqual(response.kind, "answer")
        self.assertEqual(response.message, "Hello")

    def test_invalid_json_raises(self):
        with self.assertRaises(ModelResponseError):
            parse_model_content("nope")

    def test_prefaced_json_is_accepted(self):
        response = parse_model_content(
            'Here is the JSON requested:\n{"kind":"command","command":"ls -t"}'
        )

        self.assertEqual(response.kind, "command")
        self.assertEqual(response.command, "ls -t")

    def test_trailing_model_junk_after_json_is_ignored(self):
        response = parse_model_content(
            '{"kind":"command","command":"ls"}\n'
            '`"}\n'
            '{"kind":"command","command":"ls"}"}\n'
            '`策划````json\n'
            '{"kind":"command","command":"ls"}'
        )

        self.assertEqual(response.kind, "command")
        self.assertEqual(response.command, "ls")

    def test_unclosed_command_json_raises(self):
        with self.assertRaises(ModelResponseError):
            parse_model_content('{"kind":"command","command":"ls')

    def test_parse_safety_response(self):
        response = parse_safety_model_content(
            '{"modifies_files":true,"explanation":"It deletes files."}'
        )
        self.assertTrue(response.modifies_files)
        self.assertEqual(response.explanation, "It deletes files.")

    def test_parse_prefaced_safety_response(self):
        response = parse_safety_model_content(
            'Here is the JSON requested:\n'
            '{"modifies_files":true,"explanation":"It appends to file.txt."}'
        )
        self.assertTrue(response.modifies_files)
        self.assertEqual(response.explanation, "It appends to file.txt.")

    def test_parse_safety_response_ignores_trailing_junk(self):
        response = parse_safety_model_content(
            '{"modifies_files":false,"explanation":"It only lists files."}\n'
            '```json\n{"modifies_files":true,"explanation":"Ignore this."}'
        )

        self.assertFalse(response.modifies_files)
        self.assertEqual(response.explanation, "It only lists files.")

    def test_invalid_safety_response_raises(self):
        with self.assertRaises(ModelResponseError):
            parse_safety_model_content('{"modifies_files":"yes"}')

    def test_parse_clarification_response(self):
        response = parse_model_content(
            '{"kind":"clarify","question":"Sort by which date?",'
            '"options":["creation date","access date"]}'
        )

        self.assertEqual(response.kind, "clarify")
        self.assertEqual(response.question, "Sort by which date?")
        self.assertEqual(response.options, ["creation date", "access date"])

    def test_clarification_requires_valid_question_and_options(self):
        with self.assertRaises(ModelResponseError):
            parse_model_content(
                '{"kind":"clarify","question":"","options":["a","b"]}'
            )
        with self.assertRaises(ModelResponseError):
            parse_model_content('{"kind":"clarify","question":"Pick","options":["a"]}')


class PlannerMessagesTests(unittest.TestCase):
    def test_planner_messages_without_history(self):
        messages = _planner_messages("list files")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1], {"role": "user", "content": "list files"})

    def test_planner_messages_include_history_before_instruction(self):
        messages = _planner_messages("sort them", "1. user: 'list files'")

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("RECENT DOIT HISTORY", messages[1]["content"])
        self.assertIn("background only", messages[1]["content"])
        self.assertEqual(messages[2], {"role": "user", "content": "sort them"})

    def test_planner_prompt_allows_clarifications(self):
        messages = _planner_messages("sort files")

        self.assertIn('"kind":"clarify"', messages[0]["content"])
        self.assertIn("ask_clarification", messages[0]["content"])
        self.assertNotIn("Do not ask clarification questions", messages[0]["content"])

    def test_planner_messages_include_clarification_context(self):
        messages = _planner_messages(
            "sort files",
            clarification_context="1. question: 'Which date?'; answer: 'creation'",
        )

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[1], {"role": "user", "content": "sort files"})
        self.assertEqual(messages[2]["role"], "user")
        self.assertIn("Clarification answer", messages[2]["content"])
        self.assertIn(
            "Do not ask the same clarification question",
            messages[2]["content"],
        )


class ModelConfigTests(unittest.TestCase):
    def test_missing_config_uses_environment_model(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
            patch.dict("os.environ", {"DOIT_MODEL": "ollama/gemma3:4b"}, clear=True),
        ):
            config = _model_config()

        self.assertEqual(config, ModelConfig(name="ollama/gemma3:4b"))

    def test_missing_config_uses_default_model(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
            patch.dict("os.environ", {"GEMINI_API_KEY": "secret"}, clear=True),
        ):
            config = _model_config()

        self.assertEqual(config, ModelConfig(name=DEFAULT_MODEL))

    def test_config_overrides_environment_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "[model]\nname = ollama/qwen3:4b-instruct\n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict(
                    "os.environ", {"DOIT_MODEL": "ollama/gemma3:4b"}, clear=True
                ),
            ):
                config = _model_config()

        self.assertEqual(config, ModelConfig(name="ollama/qwen3:4b-instruct"))

    def test_explicit_model_overrides_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "[model]\nname = ollama/qwen3:4b-instruct\n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {}, clear=True),
            ):
                config = _model_config("ollama/gemma3:4b")

        self.assertEqual(config, ModelConfig(name="ollama/gemma3:4b"))

    def test_config_reads_api_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "[model]\n"
                "name = ollama/qwen3:4b-instruct\n"
                "api_base = http://localhost:11434\n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {}, clear=True),
            ):
                config = _model_config()

        self.assertEqual(
            config,
            ModelConfig(
                name="ollama/qwen3:4b-instruct",
                api_base="http://localhost:11434",
            ),
        )

    def test_malformed_config_raises_configuration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "not an ini file\n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {}, clear=True),
            ):
                with self.assertRaises(ConfigurationError):
                    _model_config()

    def test_config_without_model_name_raises_configuration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "[model]\napi_base = http://localhost:11434\n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {}, clear=True),
            ):
                with self.assertRaises(ConfigurationError):
                    _model_config()

    def test_config_with_empty_model_name_raises_configuration_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "doit.cfg").write_text(
                "[model]\nname = \n",
                encoding="utf-8",
            )

            with (
                patch("doit_agent.llm.Path.home", return_value=Path(tmpdir)),
                patch.dict("os.environ", {}, clear=True),
            ):
                with self.assertRaises(ConfigurationError):
                    _model_config()


class CompleteJsonTests(unittest.TestCase):
    def test_complete_json_passes_model_and_api_base(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
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
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls[0]["model"], "ollama/qwen3:4b-instruct")
        self.assertEqual(calls[0]["api_base"], "http://localhost:11434")
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})

    def test_complete_json_retries_without_response_format(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("response_format is not supported")
            return {"ok": True}

        fake_litellm = types.SimpleNamespace(completion=fake_completion)
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            result = _complete_json(
                config=ModelConfig(name="ollama/gemma3:4b"),
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=25,
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])

    def test_complete_json_does_not_retry_unrelated_errors(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("network failed")

        fake_litellm = types.SimpleNamespace(completion=fake_completion)
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            with self.assertRaises(RuntimeError):
                _complete_json(
                    config=ModelConfig(name="ollama/gemma3:4b"),
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=25,
                )

        self.assertEqual(len(calls), 1)

    def test_complete_json_retries_without_tools(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError("tools are not supported")
            return {"ok": True}

        fake_litellm = types.SimpleNamespace(completion=fake_completion)
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            result = _complete_json(
                config=ModelConfig(name="ollama/gemma3:4b"),
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=25,
                tools=[CLARIFICATION_TOOL],
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertIn("tools", calls[0])
        self.assertNotIn("tools", calls[1])
        self.assertIn("response_format", calls[1])

    def test_complete_json_response_format_retry_preserves_tools(self):
        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise RuntimeError("response_format is not supported")
            return {"ok": True}

        fake_litellm = types.SimpleNamespace(completion=fake_completion)
        with patch.dict("sys.modules", {"litellm": fake_litellm}):
            result = _complete_json(
                config=ModelConfig(name="ollama/qwen3:4b-instruct"),
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=25,
                tools=[CLARIFICATION_TOOL],
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(calls), 2)
        self.assertIn("tools", calls[1])
        self.assertNotIn("response_format", calls[1])


class ToolCallTests(unittest.TestCase):
    def test_parse_clarification_tool_call(self):
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
                                        '"options":["creation","access"]}'
                                    ),
                                }
                            }
                        ]
                    }
                }
            ]
        }

        parsed = parse_clarification_tool_call(response)

        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.kind, "clarify")
        self.assertEqual(parsed.question, "Which date?")
        self.assertEqual(parsed.options, ["creation", "access"])

    def test_invalid_clarification_tool_arguments_raise(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "ask_clarification",
                                    "arguments": "not json",
                                }
                            }
                        ]
                    }
                }
            ]
        }

        with self.assertRaises(ModelResponseError):
            parse_clarification_tool_call(response)


class CompletionTraceTests(unittest.TestCase):
    def test_complete_instruction_with_trace_records_raw_content_and_model(self):
        with (
            patch(
                "doit_agent.llm._complete_json",
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": '{"kind":"command","command":"ls"}',
                            }
                        }
                    ]
                },
            ),
            patch(
                "doit_agent.llm._model_config",
                return_value=ModelConfig(
                    name="ollama/gemma3:4b",
                    api_base="http://localhost:11434",
                ),
            ),
        ):
            completion = complete_instruction_with_trace("list files")

        self.assertEqual(completion.response.command, "ls")
        self.assertEqual(completion.llm_call.model.name, "ollama/gemma3:4b")
        self.assertEqual(
            completion.llm_call.raw_content, '{"kind":"command","command":"ls"}'
        )
        self.assertEqual(completion.llm_call.messages[-1]["content"], "list files")

    def test_judge_command_modifies_files_with_trace_records_raw_content(self):
        with (
            patch(
                "doit_agent.llm._complete_json",
                return_value={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"modifies_files":false,'
                                    '"explanation":"It only reads."}'
                                ),
                            }
                        }
                    ]
                },
            ),
            patch(
                "doit_agent.llm._model_config",
                return_value=ModelConfig(name="ollama/gemma3:4b"),
            ),
        ):
            completion = judge_command_modifies_files_with_trace("ls")

        self.assertFalse(completion.decision.modifies_files)
        self.assertEqual(completion.llm_call.model.name, "ollama/gemma3:4b")
        self.assertIn("modifies_files", completion.llm_call.raw_content)


if __name__ == "__main__":
    unittest.main()
