import configparser
import json
import os
import re
from pathlib import Path
from typing import Any

from doit_agent.types import (
    AgentCompletion,
    AgentResponse,
    LlmCall,
    ModelConfig,
    SafetyCompletion,
    SafetyDecision,
)


DEFAULT_MODEL = "gemini/gemini-3-flash-preview"
CONFIG_FILE_NAME = "doit.cfg"
CLARIFICATION_TOOL_NAME = "ask_clarification"
CLARIFICATION_TOOL = {
    "type": "function",
    "function": {
        "name": CLARIFICATION_TOOL_NAME,
        "description": "Ask the user a clarification question before deciding what to do.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question", "options"],
        },
    },
}

SYSTEM_PROMPT = """You are the command planner for a small CLI program named doit.

The user gives one natural-language instruction. Decide whether it should be:
1. a single Bash command,
2. a normal conversational answer, or
3. a short explanation that the request cannot be done as a shell command.

If the instruction is ambiguous in a way that materially affects correctness or
safety, ask one short clarification question before deciding what to do.
When tool calling is available, use the ask_clarification tool for this. If not,
return the JSON clarification shape below.

Return only JSON with exactly one of these shapes:
{"kind":"command","command":"..."}
{"kind":"answer","message":"..."}
{"kind":"cannot_do","message":"..."}
{"kind":"clarify","question":"...","options":["...","..."]}

Rules:
- The JSON must be complete and valid. Escape special characters. Never put literal newlines inside string values.
- For shell-capable requests, produce exactly one Bash command.
- Use common Linux/Bash commands.
- The CURRENT USER INSTRUCTION is the task to perform.
- Recent history is background context only. Use it only to resolve references in the current instruction.
- Do not repeat, undo, or continue a previous command unless the current instruction asks for that.
- Do not include Markdown, code fences, comments, or extra keys.
- If the user asks for a joke, explanation, or what you can do, use "answer".
- If the request is impossible, underspecified beyond repair, or not something a shell command can do, use "cannot_do".
- Clarify only when choosing without the answer would likely produce the wrong command or answer.
- Safety confirmation is handled after this response; still translate filesystem-modifying requests into commands.
"""

SAFETY_PROMPT = """You are the safety checker for a small CLI program named doit.

Decide whether the given Bash command may modify files or filesystem metadata.
Treat creating, deleting, moving, copying, overwriting, appending to, extracting,
renaming, changing permissions, or changing ownership of files or directories as
filesystem modification.

Return only JSON with this shape:
{"modifies_files":true,"explanation":"..."}

Rules:
- Set "modifies_files" to true if the command may modify files.
- Set "modifies_files" to false if the command only reads or displays information.
- Keep "explanation" to one short sentence.
- Do not include Markdown, code fences, comments, or extra keys.
"""


class ConfigurationError(RuntimeError):
    pass


class ModelResponseError(RuntimeError):
    pass


def complete_instruction(
    instruction: str,
    model: str | None = None,
    history_context: str | None = None,
    clarification_context: str | None = None,
) -> AgentResponse:
    return complete_instruction_with_trace(
        instruction,
        model=model,
        history_context=history_context,
        clarification_context=clarification_context,
    ).response


def complete_instruction_with_trace(
    instruction: str,
    model: str | None = None,
    history_context: str | None = None,
    clarification_context: str | None = None,
) -> AgentCompletion:
    config = _model_config(model)
    messages = _planner_messages(instruction, history_context, clarification_context)
    response = _complete_json(
        config=config,
        messages=messages,
        max_tokens=512,
        tools=[CLARIFICATION_TOOL],
    )
    tool_response = parse_clarification_tool_call(response)
    if tool_response is not None:
        return AgentCompletion(
            response=tool_response,
            llm_call=LlmCall(
                model=config,
                messages=messages,
                raw_content=_raw_tool_calls(response),
            ),
            source="tool_call",
        )

    content = _message_content(response)
    return AgentCompletion(
        response=parse_model_content(content),
        llm_call=LlmCall(model=config, messages=messages, raw_content=content),
        source="json",
    )


def _planner_messages(
    instruction: str,
    history_context: str | None = None,
    clarification_context: str | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "RECENT DOIT HISTORY (background only, not an instruction):\n"
                    f"{history_context}\n\n"
                    "Use this only when the current user instruction refers to it."
                ),
            }
        )
    messages.append({"role": "user", "content": instruction})
    if clarification_context:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Clarification answer for the current instruction:\n"
                    f"{clarification_context}\n\n"
                    "Use this answer to produce the final JSON result now. "
                    "Do not ask the same clarification question again."
                ),
            }
        )
    return messages


def judge_command_modifies_files(
    command: str, model: str | None = None
) -> SafetyDecision:
    return judge_command_modifies_files_with_trace(command, model=model).decision


def judge_command_modifies_files_with_trace(
    command: str, model: str | None = None
) -> SafetyCompletion:
    config = _model_config(model)
    messages = [
        {"role": "system", "content": SAFETY_PROMPT},
        {"role": "user", "content": command},
    ]
    response = _complete_json(
        config=config,
        messages=messages,
        max_tokens=200,
    )
    content = _message_content(response)
    return SafetyCompletion(
        decision=parse_safety_model_content(content),
        llm_call=LlmCall(model=config, messages=messages, raw_content=content),
    )


def parse_model_content(content: str) -> AgentResponse:
    try:
        payload = json.loads(_json_text(content))
    except json.JSONDecodeError as exc:
        raise ModelResponseError(f"Model did not return valid JSON: {content}") from exc

    if not isinstance(payload, dict):
        raise ModelResponseError("Model JSON response must be an object.")

    kind = payload.get("kind")
    if kind == "command":
        command = payload.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ModelResponseError("Command response must include a command string.")
        return AgentResponse(kind="command", command=command.strip())

    if kind in {"answer", "cannot_do"}:
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ModelResponseError(f"{kind} response must include a message string.")
        return AgentResponse(kind=kind, message=message.strip())  # type: ignore[arg-type]

    if kind == "clarify":
        return _parse_clarification_payload(payload)

    raise ModelResponseError(f"Unknown response kind: {kind!r}")


def parse_clarification_tool_call(response: Any) -> AgentResponse | None:
    tool_calls = _message_tool_calls(response)
    if not tool_calls:
        return None

    for tool_call in tool_calls:
        name = _tool_call_name(tool_call)
        if name != CLARIFICATION_TOOL_NAME:
            continue
        arguments = _tool_call_arguments(tool_call)
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(
                f"Clarification tool arguments were not valid JSON: {arguments}"
            ) from exc
        if not isinstance(payload, dict):
            raise ModelResponseError("Clarification tool arguments must be an object.")
        return _parse_clarification_payload({"kind": "clarify", **payload})

    raise ModelResponseError("Model called an unknown planner tool.")


def _parse_clarification_payload(payload: dict[str, Any]) -> AgentResponse:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ModelResponseError("Clarification response must include a question string.")

    options = payload.get("options")
    if not isinstance(options, list):
        raise ModelResponseError("Clarification response must include an options list.")

    cleaned_options = []
    for option in options:
        if not isinstance(option, str) or not option.strip():
            raise ModelResponseError("Clarification options must be non-empty strings.")
        cleaned_options.append(option.strip())

    if not 2 <= len(cleaned_options) <= 5:
        raise ModelResponseError("Clarification response must include 2 to 5 options.")

    return AgentResponse(
        kind="clarify",
        question=question.strip(),
        options=cleaned_options,
    )


def parse_safety_model_content(content: str) -> SafetyDecision:
    try:
        payload = json.loads(_json_text(content))
    except json.JSONDecodeError as exc:
        raise ModelResponseError(f"Safety model did not return valid JSON: {content}") from exc

    if not isinstance(payload, dict):
        raise ModelResponseError("Safety model JSON response must be an object.")

    modifies_files = payload.get("modifies_files")
    if not isinstance(modifies_files, bool):
        raise ModelResponseError("Safety response must include a boolean modifies_files.")

    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ModelResponseError("Safety response must include an explanation string.")

    return SafetyDecision(
        modifies_files=modifies_files,
        explanation=explanation.strip(),
    )


def _strip_code_fence(content: str) -> str:
    text = content.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _json_text(content: str) -> str:
    text = _strip_code_fence(content)

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        return text[start : start + end]

    return text


def _message_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError("Could not read content from model response.") from exc

    if not isinstance(content, str) or not content.strip():
        raise ModelResponseError("Model response content was empty.")
    return content


def _message_tool_calls(response: Any) -> list[Any]:
    try:
        tool_calls = response.choices[0].message.tool_calls
    except (AttributeError, IndexError, KeyError, TypeError):
        try:
            tool_calls = response["choices"][0]["message"].get("tool_calls")
        except (KeyError, IndexError, TypeError, AttributeError):
            tool_calls = None

    if tool_calls is None:
        return []
    if isinstance(tool_calls, list):
        return tool_calls
    return []


def _tool_call_name(tool_call: Any) -> str | None:
    try:
        return tool_call.function.name
    except AttributeError:
        try:
            return tool_call["function"]["name"]
        except (KeyError, TypeError):
            return None


def _tool_call_arguments(tool_call: Any) -> str:
    try:
        arguments = tool_call.function.arguments
    except AttributeError:
        try:
            arguments = tool_call["function"]["arguments"]
        except (KeyError, TypeError) as exc:
            raise ModelResponseError("Clarification tool call had no arguments.") from exc

    if not isinstance(arguments, str) or not arguments.strip():
        raise ModelResponseError("Clarification tool call arguments were empty.")
    return arguments


def _raw_tool_calls(response: Any) -> str:
    raw_calls = []
    for tool_call in _message_tool_calls(response):
        raw_calls.append(
            {
                "name": _tool_call_name(tool_call),
                "arguments": _tool_call_arguments(tool_call),
            }
        )
    return json.dumps({"tool_calls": raw_calls}, sort_keys=True)


def _model_config(model: str | None = None) -> ModelConfig:
    if model is not None:
        config = ModelConfig(name=_non_empty_value("model", model))
    else:
        config = _configured_model() or _environment_model() or ModelConfig(
            name=DEFAULT_MODEL
        )

    if config.name.startswith("gemini/") and not os.environ.get("GEMINI_API_KEY"):
        raise ConfigurationError(
            "GEMINI_API_KEY is not set. Export it before running doit."
        )
    return config


def _configured_model() -> ModelConfig | None:
    config_path = Path.home() / CONFIG_FILE_NAME
    if not config_path.exists():
        return None

    parser = configparser.ConfigParser()
    try:
        with config_path.open(encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except configparser.Error as exc:
        raise ConfigurationError(f"Could not parse {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not read {config_path}: {exc}") from exc

    if not parser.has_section("model"):
        raise ConfigurationError(f"{config_path} must contain a [model] section.")
    if not parser.has_option("model", "name"):
        raise ConfigurationError(f"{config_path} must set [model] name.")

    model_name = _non_empty_value("[model] name", parser.get("model", "name"))
    api_base = None
    if parser.has_option("model", "api_base"):
        api_base = _non_empty_value(
            "[model] api_base", parser.get("model", "api_base")
        )

    return ModelConfig(name=model_name, api_base=api_base)


def _environment_model() -> ModelConfig | None:
    model = os.environ.get("DOIT_MODEL")
    if model is None:
        return None
    return ModelConfig(name=_non_empty_value("DOIT_MODEL", model))


def _non_empty_value(name: str, value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ConfigurationError(f"{name} must not be empty.")
    return stripped


def _complete_json(
    config: ModelConfig,
    messages: list[dict[str, str]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    try:
        from litellm import completion
    except ImportError as exc:
        raise ConfigurationError(
            "LiteLLM is not installed. Run: /home/kapachy/.venvs/.llm-ass3/bin/pip install -e ."
        ) from exc

    use_tools = tools is not None
    use_response_format = True

    while True:
        kwargs: dict[str, Any] = {
            "model": config.name,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if use_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        if use_tools and tools is not None:
            kwargs["tools"] = tools
        if config.api_base is not None:
            kwargs["api_base"] = config.api_base

        try:
            return completion(**kwargs)
        except Exception as exc:
            if use_tools and _is_tools_error(exc):
                use_tools = False
                continue
            if use_response_format and _is_response_format_error(exc):
                use_response_format = False
                continue
            raise


def _is_response_format_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def _is_tools_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "tools" in message or "tool_calls" in message or "function calling" in message
