import subprocess

from doit_agent.types import ShellResult


def run_shell(command: str, shell: str = "/bin/bash", timeout: int = 20) -> ShellResult:
    try:
        result = subprocess.run(
            command,
            shell=True,
            executable=shell,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return ShellResult(
            stdout=exc.stdout or "",
            stderr=f"Command timed out after {timeout} seconds.\n",
            returncode=124,
        )

    return ShellResult(
        stdout=result.stdout,
        stderr=result.stderr,
        returncode=result.returncode,
    )

