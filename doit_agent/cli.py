import argparse
import sys

from doit_agent.history import (
    append_history,
    append_report_history,
    format_history,
    load_recent_history,
)
from doit_agent.llm import (
    ConfigurationError,
    ModelResponseError,
    complete_instruction_with_trace,
)
from doit_agent.safety import assess_command_safety_with_trace
from doit_agent.shell import run_shell
from doit_agent.types import (
    AgentCompletion,
    AgentResponse,
    LlmCall,
    ModelConfig,
    SafetyAssessment,
)


AGENT_VERSION = "part5_clarifications"
REPORT_TEXT_LIMIT = 4000
MAX_CLARIFICATIONS = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doit",
        description="Translate one natural-language request into a Bash command.",
    )
    parser.add_argument("instruction", nargs="+", help="natural-language instruction")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    instruction = " ".join(args.instruction).strip()
    history_context = format_history(load_recent_history())
    report = {
        "agent_version": AGENT_VERSION,
        "instruction": instruction,
        "planner_calls": [],
    }
    clarifications: list[dict[str, object]] = []

    try:
        response = _plan_with_clarifications(
            instruction, history_context, clarifications, report
        )
        if response is None:
            return 0
    except ConfigurationError as exc:
        _append_report_error(report, "configuration", exc)
        print(f"doit: {exc}", file=sys.stderr)
        return 2
    except ModelResponseError as exc:
        _append_report_error(report, "planner_response", exc)
        print(f"doit: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        _append_report_error(report, "planner_call", exc)
        print(f"doit: model call failed: {exc}", file=sys.stderr)
        return 1

    if response.kind == "command":
        assert response.command is not None
        print(response.command, flush=True)
        try:
            safety_assessment = assess_command_safety_with_trace(response.command)
            safety = safety_assessment.decision
            report["safety"] = _safety_assessment_dict(safety_assessment)
        except ConfigurationError as exc:
            _append_report_error(report, "safety_configuration", exc)
            print(f"doit: {exc}", file=sys.stderr)
            return 2
        except ModelResponseError as exc:
            _append_report_error(report, "safety_response", exc)
            print(f"doit: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:
            _append_report_error(report, "safety_call", exc)
            print(f"doit: safety check failed: {exc}", file=sys.stderr)
            return 1

        if safety.modifies_files:
            print(f"doit: {safety.explanation}")
            try:
                confirmation = input("Proceed? [y/N] ")
            except EOFError:
                confirmation = ""

            if confirmation.strip().lower() != "y":
                print("Aborted.")
                report["confirmation"] = {
                    "required": True,
                    "response": confirmation,
                    "accepted": False,
                }
                report["execution"] = {"executed": False, "returncode": None}
                append_history(
                    {
                        "instruction": instruction,
                        "kind": response.kind,
                        "command": response.command,
                        "executed": False,
                        "returncode": None,
                        **(
                            {"clarifications": clarifications}
                            if clarifications
                            else {}
                        ),
                    }
                )
                append_report_history(report)
                return 0
            report["confirmation"] = {
                "required": True,
                "response": confirmation,
                "accepted": True,
            }
        else:
            report["confirmation"] = {"required": False}

        result = run_shell(response.command)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        report["execution"] = {
            "executed": True,
            "command": response.command,
            "stdout": _truncate(result.stdout),
            "stderr": _truncate(result.stderr),
            "returncode": result.returncode,
        }
        append_history(
            {
                "instruction": instruction,
                "kind": response.kind,
                "command": response.command,
                "executed": True,
                "returncode": result.returncode,
                **({"clarifications": clarifications} if clarifications else {}),
            }
        )
        append_report_history(report)
        return result.returncode

    assert response.message is not None
    print(response.message)
    report["execution"] = {"executed": False, "returncode": None}
    append_history(
        {
            "instruction": instruction,
            "kind": response.kind,
            "message": response.message,
            "executed": False,
            "returncode": None,
            **({"clarifications": clarifications} if clarifications else {}),
        }
    )
    append_report_history(report)
    return 0


def _plan_with_clarifications(
    instruction: str,
    history_context: str | None,
    clarifications: list[dict[str, object]],
    report: dict[str, object],
) -> AgentResponse | None:
    for _ in range(MAX_CLARIFICATIONS):
        completion = complete_instruction_with_trace(
            instruction,
            history_context=history_context,
            clarification_context=_format_clarifications(clarifications),
        )
        response = completion.response
        _record_planner_call(report, completion)
        if response.kind != "clarify":
            return response
        if any(
            clarification.get("question") == response.question
            and clarification.get("options") == response.options
            for clarification in clarifications
        ):
            print("Aborted.")
            report["error"] = {
                "stage": "clarification",
                "type": "RepeatedClarification",
                "message": (
                    "The model repeated a clarification question that was "
                    "already answered."
                ),
            }
            report["execution"] = {"executed": False, "returncode": None}
            append_history(
                {
                    "instruction": instruction,
                    "kind": "clarify",
                    "executed": False,
                    "returncode": None,
                    "clarifications": clarifications,
                }
            )
            append_report_history(report)
            return None

        clarification = _ask_clarification(response)
        if not clarification["accepted"]:
            print("Aborted.")
            all_clarifications = [*clarifications, clarification]
            report["clarifications"] = all_clarifications
            report["execution"] = {"executed": False, "returncode": None}
            append_history(
                {
                    "instruction": instruction,
                    "kind": "clarify",
                    "executed": False,
                    "returncode": None,
                    "clarifications": all_clarifications,
                }
            )
            append_report_history(report)
            return None
        clarifications.append(clarification)
        report["clarifications"] = clarifications

    print("Aborted.")
    report["error"] = {
        "stage": "clarification",
        "type": "ClarificationLimit",
        "message": "Too many clarification questions.",
    }
    report["execution"] = {"executed": False, "returncode": None}
    append_history(
        {
            "instruction": instruction,
            "kind": "clarify",
            "executed": False,
            "returncode": None,
            "clarifications": clarifications,
        }
    )
    append_report_history(report)
    return None


def _record_planner_call(
    report: dict[str, object], completion: AgentCompletion
) -> None:
    planner_call = {
        "source": completion.source,
        "llm_call": _llm_call_dict(completion.llm_call),
        "parsed_response": _agent_response_dict(completion.response),
    }
    planner_calls = report.setdefault("planner_calls", [])
    assert isinstance(planner_calls, list)
    planner_calls.append(planner_call)


def _ask_clarification(response: AgentResponse) -> dict[str, object]:
    assert response.question is not None
    assert response.options is not None
    print(response.question)
    for index, option in enumerate(response.options, start=1):
        print(f"{index}. {option}")

    try:
        answer = input("> ")
    except EOFError:
        answer = ""

    stripped = answer.strip()
    base: dict[str, object] = {
        "question": response.question,
        "options": response.options,
        "answer": answer,
        "accepted": False,
    }
    if not stripped:
        return base

    selected = stripped
    if stripped.isdigit():
        option_index = int(stripped) - 1
        if option_index < 0 or option_index >= len(response.options):
            return base
        selected = response.options[option_index]

    return {**base, "accepted": True, "selected": selected}


def _format_clarifications(clarifications: list[dict[str, object]]) -> str | None:
    if not clarifications:
        return None
    lines = []
    for index, clarification in enumerate(clarifications, start=1):
        lines.append(
            f"{index}. question: {clarification['question']!r}; "
            f"answer: {clarification['selected']!r}"
        )
    return "\n".join(lines)


def _append_report_error(
    report: dict[str, object], stage: str, exc: Exception
) -> None:
    report["error"] = {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc),
    }
    append_report_history(report)


def _agent_response_dict(response: AgentResponse) -> dict[str, object]:
    payload: dict[str, object] = {"kind": response.kind}
    if response.command is not None:
        payload["command"] = response.command
    if response.message is not None:
        payload["message"] = response.message
    if response.question is not None:
        payload["question"] = response.question
    if response.options is not None:
        payload["options"] = response.options
    return payload


def _safety_assessment_dict(assessment: SafetyAssessment) -> dict[str, object]:
    payload: dict[str, object] = {
        "source": assessment.source,
        "decision": {
            "modifies_files": assessment.decision.modifies_files,
            "explanation": assessment.decision.explanation,
        },
    }
    if assessment.llm_call is not None:
        payload["llm_call"] = _llm_call_dict(assessment.llm_call)
    return payload


def _llm_call_dict(call: LlmCall) -> dict[str, object]:
    return {
        "model": _model_config_dict(call.model),
        "user_input": _user_input(call),
        "raw_content": call.raw_content,
    }


def _model_config_dict(config: ModelConfig) -> dict[str, object]:
    return {"name": config.name, "api_base": config.api_base}


def _truncate(text: str) -> str:
    if len(text) <= REPORT_TEXT_LIMIT:
        return text
    return text[:REPORT_TEXT_LIMIT] + "\n[truncated]"


def _user_input(call: LlmCall) -> str | None:
    for message in reversed(call.messages):
        if message.get("role") == "user":
            return message.get("content")
    return None


if __name__ == "__main__":
    raise SystemExit(main())
