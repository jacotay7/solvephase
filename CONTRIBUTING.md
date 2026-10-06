# Contributing to solvephase

solvephase is a numerical optics library. A contribution is complete when its
behaviour, units, tests, documentation and performance implications are all
clear. Read [AGENTS.md](https://github.com/jacotay7/solvephase/blob/main/AGENTS.md)
for the conventions and contracts; they apply to human contributors too.

## Development setup

```bash
git clone https://github.com/jacotay7/solvephase
cd solvephase
python -m pip install -e ".[dev,docs,interop,validation]"
```

For GPU work, also install the CuPy extra that matches your driver
(`.[cuda12]` or `.[cuda13]`).

## Quality gate

```bash
ruff check .
ruff format --check .
python -m mypy
python -m pytest -q --cov=solvephase
python -m pytest -q --run-slow -m slow
python -m pytest -q --run-gpu -m gpu        # needs a CUDA device
python -m pytest -q -m interop              # needs pyturb, getframes, hcipy
python validation/validate.py --quick --output /tmp/validation
python benchmarks/run.py --quick
mkdocs build --strict
python -m build
```

## What a change needs

- **New algorithm:** a module under `src/solvephase/algorithms/`, primary
  references in its docstring, tests that assert recovery accuracy (not just
  that it runs), a GPU parity test, a guide page, a benchmark case and a
  CHANGELOG entry.
- **New operator:** an exact adjoint and an adjoint test.
- **New gradient:** a finite-difference test.
- **Performance change:** before/after numbers from `benchmarks/run.py`,
  with the hardware.

## Releases

1. Bump `src/solvephase/__about__.py`.
2. Date the CHANGELOG section.
3. Merge to `main`.
4. Publish a GitHub release whose tag is the bare version (e.g. `0.2.0`).

The release workflow builds the distributions and uploads them to PyPI with
trusted publishing.
