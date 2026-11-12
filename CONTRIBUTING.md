# Contributing to mlxvs

## Build and test

C++ library, tests and benchmarks (Homebrew's MLX):

```bash
brew install mlx cmake
cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/homebrew
cmake --build build
ctest --test-dir build      # exact k-NN vs a CPU reference; graph recall vs exact k-NN; the C API
```

Run the tests under Metal's API and shader validation, as CI does, when you change a kernel: it catches
out-of-bounds accesses and threadgroup memory over Metal's 32 KB limit, which some GPUs tolerate and others do not.

```bash
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 ctest --test-dir build --output-on-failure
```

Python bindings (built against the interpreter's `mlx` wheel, so one process holds one `libmlx`):

```bash
pip install . pytest numpy
pytest python/tests
```

CI (`.github/workflows/ci.yml`) runs both on `macos-15` Apple-silicon runners, plus the format checks.

## Code style

MLX's, so that code can move between the two projects:

- C++: MLX's `.clang-format` (2-space indent, 80 columns), `// Copyright © 2026 mikepapadim` header lines,
  `[mlxvs::op] Message.` error strings, every public operation taking a trailing `mx::StreamOrDevice s = {}`.
- Python: ruff (`ruff check`, `ruff format`).
- `pre-commit install` runs all of it before each commit.

How the kernels are organised is in [docs/DESIGN.md](docs/DESIGN.md).

## Pull requests

Open an issue first for anything beyond a small fix. Include tests (C++ in `tests/`, Python in `python/tests/`) and,
for a performance change, before/after numbers from `bench/` with the machine and whether it was on power.
