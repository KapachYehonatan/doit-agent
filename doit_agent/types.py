from dataclasses import dataclass
from typing import Literal


ResponseKind = Literal["command", "answer", "cannot_do", "clarify"]


@dataclass(frozen=True)
class AgentResponse:
    kind: ResponseKind
    command: str | None = None
    message: str | None = None
    question: str | None = None
    options: list[str] | None = None


@dataclass(frozen=True)
class ShellResult:
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class SafetyDecision:
    modifies_files: bool
    explanation: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_base: str | None = None


@dataclass(frozen=True)
class LlmCall:
    model: ModelConfig
    messages: list[dict[str, str]]
    raw_content: str


@dataclass(frozen=True)
class AgentCompletion:
    response: AgentResponse
    llm_call: LlmCall
    source: str = "json"


@dataclass(frozen=True)
class SafetyCompletion:
    decision: SafetyDecision
    llm_call: LlmCall


@dataclass(frozen=True)
class SafetyAssessment:
    decision: SafetyDecision
    source: str
    llm_call: LlmCall | None = None
