# working rules

## Rules
- **Never commit, push unless the user explicitly asks.** The user commits. You can eventually suggest to create branches or worktrees.
- One small step at a time (diff under ~300 lines), with tests, then stop for the user's review.
- Use Google format for docstrings:
    """
    This is an example of Google style.

    Args:
        param1: This is the first param.
        param2: This is a second param.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.
    """ 
- Do not leave in the comments of the codebase, specific references to the repository we are
  using as references. Neither to the actual specifics of the hardware on which we are
  developing the repo. 


## Simplicity rules
- Hydra `_target_` is the only factory: no registries, no type-string switches.
- No base class until two implementations need it.
- Dimensions are derived from the `variables` config in one place; no hidden config injection.
- Delete dead code instead of keeping "superseded" modules.
- Comments are short and explain *why*; benchmark numbers go in the diary, not in code or YAML.
- Split a file when it grows past ~300 lines.
- Every feature ships with a test on toy data.
- Speed decisions are taken on measured numbers (jets/s, memory), recorded in the diary.

## Commands
```
uv sync                                          # environment (torch from the cu126 index)
uv run pytest                                    # tests (CPU, toy data)
uv run ruff check . && uv run ruff format .      # lint + format
```

## Environment
- Training runs on the GPU cluster (slurm); code reaches it through git.
- SLURM queue to use for GPU tests and trainings: gpu-K80