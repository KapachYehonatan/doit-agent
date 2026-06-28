# ACDL Instructions

Source: <https://acdlang26.github.io/acdlsite/syntax-reference.html>

ACDL describes prompt templates sent to an LLM. It is not a free-form metadata
format. A valid file is made of one or more prompt specifications.

For this project, do not write comments in `.acdl` files. The official docs
describe `//` comments, but the installed VS Code extension reports errors for
them in our setup. Put explanations in Markdown or the report instead.

## Correct Shape

Use this top-level shape:

```acdl
PromptName[@T]: {
    S: SYSTEM_INSTRUCTIONS
    U: env.user_input[@T]
}
```

Do not use invented blocks like:

```acdl
context Something {
  description: "..."
  system_prompt { ... }
}
```

That is not the documented ACDL syntax.

## Roles

ACDL role messages use short role markers:

- `S:` system message
- `U:` user message
- `A:` assistant message
- `T:` tool result message
- `N:` single completion-format block

Use single-line form for one item:

```acdl
U: env.user_input[@T]
```

Use braces for multiple items or control flow:

```acdl
S: {
    SYSTEM_INSTRUCTIONS
    OUTPUT_FORMAT
}
```

Do not nest role messages inside role messages.

## Content Elements

Use these forms inside role messages:

- Templates: `SYSTEM_INSTRUCTIONS`, `OUTPUT_FORMAT`
- Context variables: `env.user_input[@T]`, `sys.model_config`
- Response variables: `resp.command_plan[@T]`
- Prompt references: `prompt.History[@T-1]`
- Functions: `formatCommand(resp.command_plan[@T])`

Templates are `ALL_CAPS`. Functions are `camelCase`. Context variables use one
of the official namespaces: `env`, `sys`, `resp`, or `prompt`.

## Time

Use `@T` for the current turn. A previous turn is `@T-1`. Lowercase `@t` is for
iteration.

```acdl
ForEach(@t: range(1, @T-1)) {
    U: env.user_input[@t]
    A: resp.answer[@t]
}
```

For this assignment, each `doit "..."` invocation is one turn.

## Conditionals

Use documented control-flow blocks:

```acdl
If resp.command_plan[@T].kind == "command" {
    U: env.generated_command[@T]
}
Else {
    A: resp.message[@T]
}
```

`If`, `ElseIf`, `Else`, `ForEach`, and `Switch` may appear at top level or
inside a braced role message.

## How To Document `doit`

Document prompt templates, not every command a user types.

For the current implementation, useful ACDL specs are:

- `CommandPlanner[@T]`: `SYSTEM_PROMPT` plus the CLI instruction.
- `SafetyJudge[@T]`: `SAFETY_PROMPT` plus the generated shell command.
- Later, `HistoryAwarePlanner[@T]`: planner prompt plus previous turns.

Example:

```acdl
CommandPlanner[@T]: {
    S: {
        COMMAND_PLANNER_INSTRUCTIONS
        COMMAND_PLANNER_OUTPUT_FORMAT
    }
    U: env.user_instruction[@T]
}

SafetyJudge[@T]: {
    S: {
        SAFETY_CHECK_INSTRUCTIONS
        SAFETY_CHECK_OUTPUT_FORMAT
    }
    U: resp.generated_command[@T]
}
```

Configuration such as `~/doit.cfg`, chosen model, or whether the command was
confirmed by the user is important for the report, but it is not itself a role
message unless that data is included in an LLM prompt.

## VS Code Extension

The installed ACDL extension should highlight `.acdl` files, report syntax
diagnostics, and expose `ACDL: Show Preview`. Use it as a parser sanity check
after editing `.acdl` files. If the extension disagrees with the website, keep
the `.acdl` file simple enough for the extension and move notes to Markdown.
