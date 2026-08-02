---
name: Testing SynapticDistill.jl README boundary changes
---
# Testing SynapticDistill.jl README boundary changes

## When to use this skill

Use this skill when validating PRs against `SynapticDistill.jl` that touch the README ownership boundary with `plasticity-lab`, or when running the Julia test suite for this repo.

## Dev environment

- Julia is installed via `juliaup`. The repo expects the **1.12** channel (`Manifest.toml` pins `julia_version = "1.12.x"`).
- Do **not** test on Julia 1.11; the pinned `PrecompileTools` uses `Base.StaticData` internals that fail on 1.11.

## Standard commands

```bash
# From the repo root
cd /path/to/SynapticDistill-jl

# Install / instantiate dependencies (slow first time due to Zygote/NNlib)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the full test suite
julia --project=. -e 'using Pkg; Pkg.test()'

# Smoke test
julia --project=. -e 'using SynapticDistill; println("OK")'
```

## Cross-repo README boundary checks

- The canonical `plasticity-lab` checkout is usually a sibling directory: `../plasticity-lab`.
- For README-only changes that add the `Boundary with plasticity-lab (Linear LIM-25)` section, verify:
  1. The same three substantive bullets appear in `plasticity-lab/README.md` under `### Boundary with SynapticDistill.jl (Linear LIM-25)`.
  2. The mutual-denial line matches exactly: `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.
  3. External links (`https://github.com/Limen-Neural/plasticity-lab#scope-and-ownership-boundaries` and `https://linear.app/rpd-34/issue/LIM-25/...`) return HTTP 200.

## Common gotchas

- `Pkg.test()` may warn that "the project dependencies or compat requirements have changed since the manifest was last resolved." This warning does not fail the suite, but running `Pkg.resolve()` (and committing the refreshed `Manifest.toml` in a separate change) will silence it.
- If `Pkg.test()` is backgrounded because precompilation is slow, re-attach with `get_output` and a long timeout rather than restarting.

## Devin Secrets Needed

None.
