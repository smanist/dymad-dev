# Codex Instructions

After any code edit, run the relevant static checks before finishing.

For Python changes, this repo requires:

- `make lint`
- `make typecheck`

Do not report the task as complete unless both commands pass. If a check fails, either fix the issue or clearly report the blocking failure.

Prefer `make check` when both checks should run together.
