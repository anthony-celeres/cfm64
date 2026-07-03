# Contributing to CFM64

CFM64 is developed with a lightweight, protected-`main` workflow. `main` is
always meant to be installable and green across macOS, Windows, and Linux —
it is the branch a thesis committee (and, later, PyPI) can rely on.

## Git workflow

- **Never commit or push directly to `main`.** All changes land via a pull
  request from a short-lived branch.
- **One branch per unit of work**, named after the work — tests ship in the
  *same* branch as the code they cover (there is no standing `test` branch):
  - `fix/loader-buffer-sizing`
  - `feat/fair-baseline-loader`
  - `test/loader-exactly-once`
  - `ci/os-matrix-and-guardrails`
- **Milestones are tags, not branches.** Freeze what you presented:
  ```bash
  git tag -a v0.1-proposal-defense -m "State presented at proposal defense"
  git push origin v0.1-proposal-defense
  ```
  Optionally publish the tag as a GitHub Release and attach the manuscript PDF.

### The per-task cycle

```bash
git switch main && git pull
git switch -c fix/my-change
# ...edit code AND add/adjust its tests together...
git push -u origin fix/my-change
gh pr create --base main --fill
# CI runs on all 3 OSes → merge when `ci-success` is green → delete branch
```

## CI gate

Every PR runs the test suite on **macOS + Windows + Linux** (Python 3.9 and
3.12), building the pybind11 native extension from source on each. Branch
protection requires the `ci-success` check to pass, so **a PR cannot merge to
`main` while any OS fails**.

Run the same checks locally before pushing:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ".[dev]"
pytest
```

## Enable the local guardrail hook (once per clone)

Mirrors the server-side rules so mistakes are caught before they leave your
machine (blocks direct `main` pushes and AI co-author trailers):

```bash
git config core.hooksPath .githooks
```

## Commit messages

- Write clear, self-authored commit messages.
- **Do not** add `Co-Authored-By` / AI-attribution trailers.
