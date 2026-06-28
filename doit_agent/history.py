import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_LIMIT = 5
HISTORY_SCAN_LIMIT = 50
OTHER_SESSION_LIMIT = 3
SHELL_HISTORY_LIMIT = 8
SHELL_COMMAND_MAX_LENGTH = 200
REPORT_HISTORY_FILE = "report_history.jsonl"
ERROR_HISTORY_FILE = "error_history.jsonl"
MEMORY_FILE = "memories.jsonl"
SESSION_ENV = "DOIT_SESSION_ID"
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z_]*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASS|KEY)[A-Z_]*)="
    r"(\"[^\"]*\"|'[^']*'|\S+)"
)


def history_path() -> Path:
    return Path.home() / ".doit" / "history.jsonl"


def report_history_path() -> Path:
    return Path.home() / ".doit" / REPORT_HISTORY_FILE


def error_history_path() -> Path:
    return Path.home() / ".doit" / ERROR_HISTORY_FILE


def memory_path() -> Path:
    return Path.home() / ".doit" / MEMORY_FILE


def shell_history_path() -> Path:
    histfile = os.environ.get("HISTFILE")
    if histfile:
        return Path(histfile).expanduser()

    shell = os.path.basename(os.environ.get("SHELL", ""))
    if shell == "zsh":
        return Path.home() / ".zsh_history"
    return Path.home() / ".bash_history"


def current_session(cwd: str | None = None) -> dict[str, str]:
    session_id = os.environ.get(SESSION_ENV, "").strip()
    if session_id:
        return {"id": session_id, "source": "env"}

    cwd = cwd or os.getcwd()
    return {"id": f"cwd:{cwd}", "source": "cwd"}


def load_recent_history(
    limit: int = HISTORY_LIMIT, path: Path | None = None
) -> list[dict[str, Any]]:
    path = path or history_path()
    if not path.exists():
        return []

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries[-limit:]


def append_history(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    session = current_session(cwd)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": cwd,
        "session_id": session["id"],
        **entry,
    }
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, sort_keys=True) + "\n")


def append_report_history(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or report_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    session = current_session(cwd)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": cwd,
        "session_id": session["id"],
        **entry,
    }
    with path.open("a", encoding="utf-8") as report_file:
        report_file.write(json.dumps(record, sort_keys=True) + "\n")


def append_error_history(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or error_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    session = current_session(cwd)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": cwd,
        "session_id": session["id"],
        **entry,
    }
    with path.open("a", encoding="utf-8") as error_file:
        error_file.write(json.dumps(record, sort_keys=True) + "\n")


def load_memories(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or memory_path()
    if not path.exists():
        return []

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("memory"), str):
            entries.append(entry)
    return entries


def append_memories(
    memories: list[str],
    instruction: str,
    path: Path | None = None,
) -> None:
    path = path or memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    session = current_session(cwd)
    with path.open("a", encoding="utf-8") as memory_file:
        for memory in memories:
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cwd": cwd,
                "session_id": session["id"],
                "memory": memory,
                "source_instruction": instruction,
            }
            memory_file.write(json.dumps(record, sort_keys=True) + "\n")


def load_recent_shell_history(
    limit: int = SHELL_HISTORY_LIMIT,
    path: Path | None = None,
) -> list[str]:
    path = path or shell_history_path()
    if not path.exists():
        return []

    commands = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        command = _parse_shell_history_line(line)
        if command and not _skip_shell_command(command):
            commands.append(_sanitize_shell_command(command))
    return _prefer_non_doit_commands(commands, limit)


def _parse_shell_history_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(": ") and ";" in stripped:
        return stripped.split(";", 1)[1].strip() or None
    return stripped


def _sanitize_shell_command(command: str) -> str:
    redacted = SECRET_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", command)
    if len(redacted) <= SHELL_COMMAND_MAX_LENGTH:
        return redacted
    return redacted[:SHELL_COMMAND_MAX_LENGTH] + "...[truncated]"


def _skip_shell_command(command: str) -> bool:
    stripped = command.strip()
    if stripped.startswith("export "):
        return True
    if re.match(r"echo\s+\$.*(?:API_KEY|TOKEN|SECRET|PASSWORD|PASS|KEY)", stripped, re.I):
        return True
    if len(stripped) > SHELL_COMMAND_MAX_LENGTH and re.match(
        r"(?:[A-Z_][A-Z0-9_]*=.*\s+){2,}", stripped
    ):
        return True
    return False


def _prefer_non_doit_commands(commands: list[str], limit: int) -> list[str]:
    recent = list(enumerate(commands))[-limit * 3 :]
    non_doit = [(index, command) for index, command in recent if not _is_doit(command)]
    selected = non_doit[-limit:]
    if len(selected) < limit:
        selected_indexes = {index for index, _ in selected}
        fill = [
            (index, command)
            for index, command in recent
            if index not in selected_indexes
        ][-(limit - len(selected)) :]
        selected.extend(fill)
    return [command for _, command in sorted(selected)]


def _is_doit(command: str) -> bool:
    return command == "doit" or command.startswith("doit ")


def format_history(entries: list[dict[str, Any]]) -> str | None:
    if not entries:
        return None

    lines = [_format_history_entry(index, entry) for index, entry in enumerate(entries, start=1)]

    return "\n".join(lines)


def format_session_history(
    entries: list[dict[str, Any]],
    session_id: str,
    cwd: str,
    current_limit: int = HISTORY_LIMIT,
    other_limit: int = OTHER_SESSION_LIMIT,
) -> str | None:
    current = []
    other = []
    for entry in entries:
        if _entry_matches_session(entry, session_id, cwd):
            current.append(entry)
        else:
            other.append(entry)

    lines = [f"CURRENT SESSION: {session_id!r}; CURRENT DIRECTORY: {cwd!r}"]
    current = current[-current_limit:]
    other = other[-other_limit:]
    if current:
        lines.append("CURRENT DOIT SESSION HISTORY:")
        lines.append(format_history(current) or "")
    if other:
        lines.append(
            "OTHER DOIT SESSIONS (use only when the user explicitly references another session/window/task):"
        )
        for index, entry in enumerate(other, start=1):
            lines.append(_format_history_entry(index, entry, include_session=True))

    return "\n".join(lines) if lines else None


def _entry_matches_session(entry: dict[str, Any], session_id: str, cwd: str) -> bool:
    entry_session = entry.get("session_id")
    if entry_session == session_id:
        return True
    return entry_session is None and entry.get("cwd") == cwd


def _format_history_entry(
    index: int,
    entry: dict[str, Any],
    include_session: bool = False,
) -> str:
    instruction = entry.get("instruction", "")
    kind = entry.get("kind", "")
    command = entry.get("command")
    message = entry.get("message")
    stdout = entry.get("stdout")
    stderr = entry.get("stderr")
    returncode = entry.get("returncode")
    executed = entry.get("executed")

    summary = f"{index}. "
    if include_session:
        summary += f"session: {entry.get('session_id')!r}; cwd: {entry.get('cwd')!r}; "
    summary += f"user: {instruction!r}; kind: {kind}"
    if command:
        summary += (
            f"; command: {command!r}; executed: {executed}; "
            f"returncode: {returncode}"
        )
    if message:
        summary += f"; message: {message!r}"
    if stdout:
        summary += f"; stdout: {stdout!r}"
    if stderr:
        summary += f"; stderr: {stderr!r}"
    return summary


def format_memories(entries: list[dict[str, Any]]) -> str | None:
    lines = []
    for index, entry in enumerate(entries, start=1):
        memory = entry.get("memory")
        if isinstance(memory, str) and memory.strip():
            lines.append(f"{index}. {memory.strip()!r}")
    return "\n".join(lines) if lines else None


def format_user_context(cwd: str, shell_commands: list[str]) -> str:
    lines = [f"CURRENT DIRECTORY: {cwd}"]
    if shell_commands:
        lines.append("RECENT USER SHELL COMMANDS:")
        lines.extend(
            f"{index}. {command!r}"
            for index, command in enumerate(shell_commands, start=1)
        )
    return "\n".join(lines)
