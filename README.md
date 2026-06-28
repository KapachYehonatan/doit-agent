# doit agent

`doit` is a small command-line LLM agent for translating natural-language
requests into Bash commands, executing them, and keeping enough context to
support follow-up requests.

This project was built for Assignment 3: Agents. It currently supports:

- single-command planning through LiteLLM
- hosted Gemini and local Ollama models via `~/doit.cfg`
- dangerous-command detection and user confirmation
- multi-turn history in `~/.doit/history.jsonl`
- user-awareness from recent shell history
- multi-terminal context separation with `DOIT_SESSION_ID`
- report-oriented logs in `~/.doit/report_history.jsonl`
- clarifying questions through tool calls or JSON fallback
- ACDL prompt documentation under `docs/acdl/`

## Install

Use the assignment virtualenv:

```bash
/home/kapachy/.venvs/.llm-ass3/bin/pip install -e .
```

After installation, the console script is:

```bash
doit "list files"
```

There is also an executable wrapper at `./doit`.

## Configure A Model

Create `~/doit.cfg`:

```ini
[model]
name = gemini/gemini-3-flash-preview
```

For Gemini, set:

```bash
export GEMINI_API_KEY=...
```

For Ollama:

```ini
[model]
name = ollama/gemma3:4b
api_base = http://localhost:11434
```

Ollama must be running separately before using an `ollama/...` model.

## Usage

```bash
doit "list files"
doit "sort them by date"
doit "create a notes folder"
```

If a command may modify files, `doit` prints the command, explains the risk,
and asks for confirmation before running it.

If the request is ambiguous, `doit` may ask a clarification question and then
continue planning after the answer.

For reliable awareness of commands typed in the current Bash session, add this
to your shell setup:

```bash
export DOIT_SESSION_ID="${DOIT_SESSION_ID:-$$_$(date +%s)}"
export PROMPT_COMMAND="history -a; history -n${PROMPT_COMMAND:+; $PROMPT_COMMAND}"
```

`DOIT_SESSION_ID` lets `doit` keep separate context streams for different
terminal windows. Without it, `doit` falls back to grouping history by current
directory. Without the `PROMPT_COMMAND` hook, Bash may not write
current-session commands to `~/.bash_history` until the terminal exits.

## State Files

Runtime state is stored outside the repository:

- `~/.doit/history.jsonl`: compact history used for multi-turn context
- `DOIT_SESSION_ID`: optional terminal-window id for multi-tasking context
- `~/.bash_history` or `$HISTFILE`: recent user shell commands used for user awareness
- `~/.doit/report_history.jsonl`: richer invocation log for the report
- `~/.doit/error_history.jsonl`: compact failure log for debugging
- `~/.doit/memories.jsonl`: durable user facts and preferences
- `~/doit.cfg`: model configuration

These files are intentionally not committed.

## Tests

Run the full suite:

```bash
/home/kapachy/.venvs/.llm-ass3/bin/python -m unittest discover -s tests
```
