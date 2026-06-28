import os
import shlex

from doit_agent.llm import judge_command_modifies_files_with_trace
from doit_agent.types import SafetyAssessment, SafetyDecision


KNOWN_MODIFYING_COMMANDS: frozenset[str] = frozenset(
    {
        "chmod",
        "chown",
        "cp",
        "install",
        "ln",
        "mkdir",
        "mv",
        "rm",
        "rmdir",
        "touch",
        "truncate",
    }
)

KNOWN_COMMAND_EXPLANATIONS = {
    "chmod": "It uses `chmod`, which can change file or directory permissions.",
    "chown": "It uses `chown`, which can change file or directory ownership.",
    "cp": "It uses `cp`, which can create or overwrite copied files.",
    "install": "It uses `install`, which can copy files and set their metadata.",
    "ln": "It uses `ln`, which can create filesystem links.",
    "mkdir": "It uses `mkdir`, which can create directories.",
    "mv": "It uses `mv`, which can move or rename files or directories.",
    "rm": "It uses `rm`, which can delete files or directories.",
    "rmdir": "It uses `rmdir`, which can delete directories.",
    "touch": "It uses `touch`, which can create files or update timestamps.",
    "truncate": "It uses `truncate`, which can change file contents or size.",
}


def assess_command_safety(command: str) -> SafetyDecision:
    return assess_command_safety_with_trace(command).decision


def assess_command_safety_with_trace(command: str) -> SafetyAssessment:
    known_command = find_known_modifying_command(command)
    if known_command is not None:
        return SafetyAssessment(
            decision=SafetyDecision(
                modifies_files=True,
                explanation=KNOWN_COMMAND_EXPLANATIONS[known_command],
            ),
            source=f"known_command:{known_command}",
        )

    completion = judge_command_modifies_files_with_trace(command)
    return SafetyAssessment(
        decision=completion.decision,
        source="llm",
        llm_call=completion.llm_call,
    )


def find_known_modifying_command(command: str) -> str | None:
    try:
        tokens = _shell_tokens(command)
    except ValueError:
        return None

    next_token_starts_command = True
    skip_next_as_command_option = False

    for token in tokens:
        if token in {";", "&&", "||", "|", "&", "(", ")"}:
            next_token_starts_command = True
            skip_next_as_command_option = False
            continue

        if token in {"sudo", "env", "command"} and next_token_starts_command:
            continue

        if token == "xargs":
            next_token_starts_command = False
            skip_next_as_command_option = False
            continue

        if token in {"sh", "bash", "zsh"} and next_token_starts_command:
            skip_next_as_command_option = token in {"sh", "bash", "zsh"}
            next_token_starts_command = False
            continue

        if skip_next_as_command_option and token in {"-c", "-lc"}:
            next_token_starts_command = True
            skip_next_as_command_option = False
            continue

        if next_token_starts_command:
            command_name = os.path.basename(token)
            if command_name in KNOWN_MODIFYING_COMMANDS:
                return command_name

        next_token_starts_command = False
        skip_next_as_command_option = False

    return None


def _shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    return list(lexer)
