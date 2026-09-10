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

- This Distill repo is [`rmems/SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl). Do not treat it as still under Limen-Neural.
- The STDP peer has **not** transferred: there is no `rmems/plasticity-lab`. The still-live checkout is [`Limen-Neural/plasticity-lab`](https://github.com/Limen-Neural/plasticity-lab) (usually a sibling directory `../plasticity-lab` when present).
- For README changes that touch the `Boundary with plasticity-lab (Linear LIM-25)` section, verify:
  1. Distill ownership reads as `rmems/SynapticDistill.jl` (not `Limen-Neural/SynapticDistill.jl`).
  2. The mutual-denial line matches exactly: `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.
  3. The plasticity-lab link stays the live external peer `https://github.com/Limen-Neural/plasticity-lab#scope-and-ownership-boundaries` (HTTP 200). Do not invent `rmems/plasticity-lab`.
  4. Linear [LIM-25](https://linear.app/rpd-34/issue/LIM-25/plasticity-lab-clarify-ownership-boundary-with-synapticdistilljl) still returns HTTP 200.

## Common gotchas

- `Pkg.test()` may warn that "the project dependencies or compat requirements have changed since the manifest was last resolved." This warning does not fail the suite, but running `Pkg.resolve()` (and committing the refreshed `Manifest.toml` in a separate change) will silence it.
- If `Pkg.test()` is backgrounded because precompilation is slow, re-attach with `get_output` and a long timeout rather than restarting.

## Devin Secrets Needed

None.
