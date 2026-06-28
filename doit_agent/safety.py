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
        "tee",
        "touch",
        "truncate",
        "unzip",
    }
)

KNOWN_READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        "awk",
        "basename",
        "cat",
        "df",
        "dirname",
        "du",
        "file",
        "find",
        "grep",
        "head",
        "ls",
        "pwd",
        "readlink",
        "realpath",
        "rg",
        "sed",
        "sort",
        "stat",
        "tail",
        "uniq",
        "wc",
        "which",
    }
)

KNOWN_COMMAND_EXPLANATIONS = {
    "chmod": "It uses `chmod`, which can change file or directory permissions.",
    "chown": "It uses `chown`, which can change file or directory ownership.",
    "cp": "It uses `cp`, which can create or overwrite copied files.",
    "find -delete": "It uses `find -delete`, which can delete files.",
    "install": "It uses `install`, which can copy files and set their metadata.",
    "ln": "It uses `ln`, which can create filesystem links.",
    "mkdir": "It uses `mkdir`, which can create directories.",
    "mv": "It uses `mv`, which can move or rename files or directories.",
    "rm": "It uses `rm`, which can delete files or directories.",
    "rmdir": "It uses `rmdir`, which can delete directories.",
    "sed -i": "It uses `sed -i`, which can modify files in place.",
    "tar -x": "It extracts an archive, which can create or overwrite files.",
    "tee": "It uses `tee`, which can write command output to files.",
    "touch": "It uses `touch`, which can create files or update timestamps.",
    "truncate": "It uses `truncate`, which can change file contents or size.",
    "unzip": "It uses `unzip`, which can create or overwrite files.",
    "redirection": "It uses output redirection, which can create or overwrite files.",
}

READ_ONLY_EXPLANATION = "It only reads or displays information."
UNKNOWN_SAFETY_EXPLANATION = (
    "Could not verify command safety automatically, so confirmation is required."
)


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
            source=f"known_modifying:{known_command}",
        )

    if is_known_read_only_command(command):
        return SafetyAssessment(
            decision=SafetyDecision(
                modifies_files=False,
                explanation=READ_ONLY_EXPLANATION,
            ),
            source="known_read_only",
        )

    try:
        completion = judge_command_modifies_files_with_trace(command)
    except Exception as exc:
        return SafetyAssessment(
            decision=SafetyDecision(
                modifies_files=True,
                explanation=UNKNOWN_SAFETY_EXPLANATION,
            ),
            source="llm_failed_assume_dangerous",
            error=str(exc),
        )

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

    redirection = _modifying_redirection(tokens)
    if redirection is not None:
        return redirection

    if _has_token_after_command(tokens, "find", "-delete"):
        return "find -delete"
    if _has_token_after_command(tokens, "sed", "-i"):
        return "sed -i"
    if _has_tar_extract(tokens):
        return "tar -x"

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


def is_known_read_only_command(command: str) -> bool:
    try:
        tokens = _shell_tokens(command)
    except ValueError:
        return False

    if find_known_modifying_command(command) is not None:
        return False
    if _has_token_after_command(tokens, "find", "-exec"):
        return False
    if _has_token_after_command(tokens, "find", "-ok"):
        return False

    found_command = False
    next_token_starts_command = True
    skip_next_as_command_option = False
    skip_next_redirection_target = False

    for index, token in enumerate(tokens):
        if skip_next_redirection_target:
            skip_next_redirection_target = False
            continue

        if token in {"<", ">", ">>"}:
            if token != "<" and _redirection_target(tokens, index) != "/dev/null":
                return False
            skip_next_redirection_target = True
            continue

        if token in {";", "&&", "||", "|", "&", "(", ")"}:
            next_token_starts_command = True
            skip_next_as_command_option = False
            continue

        if token in {"sudo", "env", "command"} and next_token_starts_command:
            continue

        if _is_env_assignment(token) and next_token_starts_command:
            continue

        if token in {"sh", "bash", "zsh"} and next_token_starts_command:
            skip_next_as_command_option = True
            next_token_starts_command = False
            continue

        if skip_next_as_command_option and token in {"-c", "-lc"}:
            return False

        if next_token_starts_command:
            command_name = os.path.basename(token)
            if command_name not in KNOWN_READ_ONLY_COMMANDS:
                return False
            found_command = True

        next_token_starts_command = False
        skip_next_as_command_option = False

    return found_command


def _shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()<>")
    lexer.whitespace_split = True
    return list(lexer)


def _modifying_redirection(tokens: list[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token in {">", ">>"} and _redirection_target(tokens, index) != "/dev/null":
            return "redirection"
    return None


def _redirection_target(tokens: list[str], index: int) -> str | None:
    try:
        return tokens[index + 1]
    except IndexError:
        return None


def _has_token_after_command(tokens: list[str], command: str, token: str) -> bool:
    in_command = False
    for item in tokens:
        if item in {";", "&&", "||", "|", "&", "(", ")"}:
            in_command = False
            continue
        if os.path.basename(item) == command:
            in_command = True
            continue
        if in_command and item == token:
            return True
    return False


def _has_tar_extract(tokens: list[str]) -> bool:
    in_tar = False
    for item in tokens:
        if item in {";", "&&", "||", "|", "&", "(", ")"}:
            in_tar = False
            continue
        if os.path.basename(item) == "tar":
            in_tar = True
            continue
        if in_tar and (item == "--extract" or item.startswith("-") and "x" in item):
            return True
    return False


def _is_env_assignment(token: str) -> bool:
    name, separator, _ = token.partition("=")
    return bool(separator and name and "/" not in name)
