# Example Script Pattern

After reviewing the runnable Python examples under `scripts/`, there are two valid example
patterns in the repo. They serve different audiences and should both remain available.

Representative current examples:

- Developer-facing if-block style:
  - `scripts/pirom_dyn/dyn_train.py`
  - `scripts/kuramoto/train.py`
  - `scripts/linear_time_invariant/lti_mp.py`
- `scripts/linear_time_invariant/lti_train_cli.py`
- `scripts/pirom_res/res_train_cli.py`
- `scripts/lorenz63/lor_train_cli.py`
- `scripts/sa_2dk/kp_sa_cli.py`

## Choose A Pattern

Use the pattern that matches the purpose of the example:

1. Use the old if-block style when the example is developer-facing and meant to expose the full
   workflow inline.
2. Use the CLI style when the example is user-facing and should be runnable without opening the
   file to change toggles or paths.

## Developer-Facing If-Block Style

This style is appropriate for examples that intentionally show the concrete execution phases in the
file itself.

Common shape:

1. Define module-level constants, configs, and case metadata near the top of the file.
2. Do not use `main()` or `if __name__ == "__main__":`.
3. Use inline phase toggles such as `ifdat`, `iftrn`, `ifviz`, `ifprd`, or `ifplt` when the goal
   is to make the execution phases visible and easy for a developer to modify.
4. Keep it acceptable for a reader to edit the file directly to switch phases, inspect
   intermediate behavior, or swap configs.
5. Whenever possible, collect possible configs of cases (e.g., combinations of models/optimizers)
   in a list at the beginning.
6. When appropriate, a typical script contains:
   - `ifdat`: Data generation
   - `iftrn`: A loop over chosen case(s) to be trained
   - `ifplt`: Plotting training history of chosen case(s)
   - `ifprd`: Prediction on test data using chosen case(s)

Use this style when showing all details is more important than providing a polished command-line
interface.

## User-Facing CLI Style

Use this shape for new user-facing runnable examples, benchmarks, and training demos under
`scripts/`:

1. Define module-level constants and case metadata near the top of the file.
2. Add a dedicated `parse_args()` function using `argparse`.
3. Reuse `scripts/cli_helpers.py` for shared concerns when the script fits that model:
   - `add_common_cli_args(...)`
   - `set_seed(...)`
   - `resolve_case_indices(...)`
   - `print_case_table(...)`
   - `stage_workdir(...)`
4. Keep each execution phase in its own function, typically:
   - `prepare_workdir(...)`
   - `generate_data(...)` when needed
   - `train(...)`
   - `plot(...)`
   - `predict(...)`
5. Keep `main()` as the only orchestration entrypoint.
6. End with an import-safe guard:

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

## CLI Main Function Responsibilities

`main()` should do the minimum coordination work needed to run the script:

1. Parse CLI arguments.
2. Apply reproducibility settings such as `set_seed(...)`.
3. Resolve the working directory from `BASE_DIR` and optional `--workdir`.
4. Stage files into the workdir if the script supports detached runs.
5. Change into the resolved working directory once if the rest of the script uses relative paths.
6. Handle `--list-cases` early and return `0`.
7. Resolve selected cases once.
8. Run optional phases based on CLI flags such as `--data`, `--no-train`, `--no-plot`,
   `--no-predict`, and `--no-show`.
9. Return an integer status code.

## CLI Recommended Skeleton

```python
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.cli_helpers import (
    add_common_cli_args,
    print_case_table,
    resolve_case_indices,
    set_seed,
    stage_workdir,
)

BASE_DIR = Path(__file__).resolve().parent

cases = [
    {"name": "example_case", "config": "example.yaml"},
]
DEFAULT_CASES = [0]


def parse_args():
    parser = argparse.ArgumentParser(description="Run example cases.")
    add_common_cli_args(parser)
    return parser.parse_args()


def prepare_workdir(root: Path):
    stage_workdir(root, BASE_DIR, ["example.yaml", "data/example.npz"], data_dir=True)


def train(selected: list[int], root: Path):
    ...


def plot(selected: list[int]):
    ...


def predict(selected: list[int]):
    ...


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    root = BASE_DIR if args.workdir is None else args.workdir.resolve()
    if args.workdir is not None:
        prepare_workdir(root)
    os.chdir(root)

    if args.list_cases:
        print_case_table(cases)
        return 0

    selected = resolve_case_indices(args.case, len(cases), DEFAULT_CASES)

    if not args.no_train:
        train(selected, root)
    if not args.no_plot:
        plot(selected)
    if not args.no_predict:
        predict(selected)
    if not args.no_show and (not args.no_plot or not args.no_predict):
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Scope Notes

- For single-purpose benchmark scripts, the same CLI `parse_args()` plus `main()` plus
  `SystemExit` pattern still applies even if there is no case table.
- If a script is only being lightly edited and already uses the if-block style, keep the edit
  narrow unless the task explicitly asks for a cleanup or a CLI conversion.
- Do not convert a developer-facing if-block example into a CLI just for consistency.
- Do not introduce inline execution toggles into a user-facing CLI example when flags are a better
  fit.
