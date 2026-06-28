import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_LIMIT = 5
REPORT_HISTORY_FILE = "report_history.jsonl"


def history_path() -> Path:
    return Path.home() / ".doit" / "history.jsonl"


def report_history_path() -> Path:
    return Path.home() / ".doit" / REPORT_HISTORY_FILE


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
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        **entry,
    }
    with path.open("a", encoding="utf-8") as history_file:
        history_file.write(json.dumps(record, sort_keys=True) + "\n")


def append_report_history(entry: dict[str, Any], path: Path | None = None) -> None:
    path = path or report_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        **entry,
    }
    with path.open("a", encoding="utf-8") as report_file:
        report_file.write(json.dumps(record, sort_keys=True) + "\n")


def format_history(entries: list[dict[str, Any]]) -> str | None:
    if not entries:
        return None

    lines = []
    for index, entry in enumerate(entries, start=1):
        instruction = entry.get("instruction", "")
        kind = entry.get("kind", "")
        command = entry.get("command")
        message = entry.get("message")
        returncode = entry.get("returncode")
        executed = entry.get("executed")

        summary = f"{index}. user: {instruction!r}; kind: {kind}"
        if command:
            summary += (
                f"; command: {command!r}; executed: {executed}; "
                f"returncode: {returncode}"
            )
        if message:
            summary += f"; message: {message!r}"
        lines.append(summary)

    return "\n".join(lines)
