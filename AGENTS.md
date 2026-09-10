# SynapticDistill.jl — Agent Instructions

**Role:** You are a Julia developer contributing to SynapticDistill, a library for modular online training of spiking neural networks (SNNs). SNNs communicate via discrete spikes rather than continuous activations. This library implements E-prop (eligibility propagation), OTTT (Online Spatio-Temporal Trace Training), and surrogate gradient methods. It supports pure SNN training and hybrid systems where an SNN is distilled from a frozen large language model (LLM).

## Project overview

- **Language:** Julia (1.8+)
- **Package name:** SynapticDistill
- **Repository:** https://github.com/rmems/SynapticDistill.jl (transferred from Limen-Neural; old org URL redirects here)
- **Wiki:** enabled under rmems — https://github.com/rmems/SynapticDistill.jl/wiki (remote `rmems/SynapticDistill.jl.wiki.git`)
- **License:** Dual MIT (Massachusetts Institute of Technology) / Apache-2.0
- **Key dependencies:** Zygote (automatic differentiation), MLUtils (machine learning utilities), LinearAlgebra

## Scope boundaries

**SynapticDistill owns:**

- Differentiable/online distillation and teacher-student transfer
- E-prop, OTTT, and surrogate gradient training rules
- Gradient computation for spiking neurons
- Training loop utilities and callbacks

**SynapticDistill does not own:**

- Reward-modulated STDP (spike-timing-dependent plasticity) or Hebbian learning — that stays in the still-live external peer [`Limen-Neural/plasticity-lab`](https://github.com/Limen-Neural/plasticity-lab) (`rmems/plasticity-lab` does not exist)
- IPC (inter-process communication) wire protocol types
- Domain-specific model architectures
- Hardware-specific optimizations (unless generic)

## Dev environment tips

```bash
# Install and instantiate
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Start a Julia REPL (Read-Eval-Print Loop) session
julia --project=.
] activate .
using SynapticDistill
```

## Build and test commands

```bash
# Run the full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Quick smoke test
julia --project=. -e 'using SynapticDistill; println("OK")'
```

## Code style

- **CRITICAL:** Every source file should begin with `# SPDX-License-Identifier: MIT OR Apache-2.0`. SPDX stands for Software Package Data Exchange, a standardized license identifier format.
- Module name: `SynapticDistill` (not `SpikenautDistill`)
- Julia naming conventions: lowercase_snake_case for functions, CamelCase for types
- Keep functions small and focused; prefer pure functions where possible

## Testing instructions

- Tests live in `test/runtests.jl`
- Test file verifies module loads and exported symbols
- Add tests for any new functions or exports
- **CRITICAL:** Run `julia --project=. -e 'using Pkg; Pkg.test()'` before committing
- CI (continuous integration) will verify tests pass automatically; aim to have them green before requesting review

## PR instructions

- Title format: `<type>: <description>` (e.g. `feat:`, `fix:`, `chore:`, `docs:`)
- New source files should include the SPDX license header
- Add or update tests for any code changes
- Run tests locally before pushing
- Resolve all review threads before requesting merge

## Cursor Cloud specific instructions

- Julia is provided via `juliaup`. The default channel is **1.12**, matching this repo's committed `Manifest.toml` (`julia_version = "1.12.x"`).
- Do not test on Julia 1.11. The pinned `PrecompileTools` references 1.12-only `Base` internals (`Base.StaticData`) and fails to precompile with `UndefVarError`.
- Run the standard commands from "Build and test commands" above:
  - `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
  - `julia --project=. -e 'using Pkg; Pkg.test()'`

- The first `instantiate`/`precompile` step is slow (~1–2 min) because of Zygote/NNlib. Later runs use the cache.
