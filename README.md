# SynapticDistill.jl

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

**Modular online training for spiking neural networks in Julia — E-prop, OTTT, and more. Works with pure SNNs or hybrid teacher-student systems.**

`SynapticDistill.jl` is a flexible and performant library for training spiking neural networks (SNNs) using online (event-based) learning rules. It is designed to be framework-agnostic, allowing researchers to bring their own models, model-step callbacks, loss functions, and data sources.

## Core Philosophy

- **Bring any model step**: The package does not own a global `forward` implementation. Callers inject a pure function or callable object that maps `(model, spikes::SpikeBatch)` to the model output used by the loss.
- **Bring any loss**: The training process is driven by a user-provided loss function. This can be a standard metric like mean squared error for pure SNN tasks, or it can be a distillation loss derived from a frozen teacher model in a hybrid setup.
- **Apply any rule**: The library provides a modular system for learning rules, starting with e-prop and OTTT. Researchers can easily add their own rules.
- **Update only the SNN**: In hybrid systems, the library is designed to update only the SNN parameters, leaving any external teacher model frozen.

## Scope and ownership boundaries

`SynapticDistill.jl` provides modular online training for spiking neural networks in Julia. It is intentionally framework-agnostic.

### Owns

- Differentiable/online distillation and teacher-student knowledge transfer
- E-prop, OTTT, and surrogate-gradient training rules
- Gradient computation for spiking neurons
- Training-loop utilities and callbacks

### Does not own

- Reward-modulated STDP or Hebbian learning
- IPC wire protocol types
- Domain-specific model architectures
- Hardware-specific optimizations (unless generic)

### Boundary with plasticity-lab (Linear LIM-25)

- `SynapticDistill.jl` (Julia): differentiable or online distillation and teacher-student knowledge transfer.
- [`plasticity-lab`](https://github.com/Limen-Neural/plasticity-lab) (Rust): reward-modulated STDP / Hebbian plasticity rules and online low-level weight delta computation.
- `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.

See the matching boundary note in the [`plasticity-lab` README](https://github.com/Limen-Neural/plasticity-lab#scope-and-ownership-boundaries) and the Linear issue [LIM-25](https://linear.app/rpd-34/issue/LIM-25/plasticity-lab-clarify-ownership-boundary-with-synapticdistilljl).

## Quick Start (Pure SNN Training)

Here's a simple example of how to train an SNN using an injected model step and a standard loss function.

```julia
using SynapticDistill
using Statistics

# 1. Define your SNN model
mutable struct MySNN
    weights::Matrix{Float32}
end

model = MySNN(rand(Float32, 10, 10))

# 2. Create a batch of spike data
spike_data = Float32.(rand(0:1, 10, 100))
spike_batch = SpikeBatch(spike_data, nothing, nothing)

# 3. Inject a model step. It can be any callable with this signature:
#    (model, spikes::SpikeBatch) -> output
function model_step(model, spikes::SpikeBatch)
    rates = vec(mean(spikes.spikes; dims=2))
    return (logits = model.weights * rates,)
end

# 4. Define a loss function over the model-step output
mse_loss(output) = sum(output.logits .^ 2)

# 5. Run a training step
model, state = train_step!(model, spike_batch, mse_loss; forward_fn=model_step, rule=:eprop)

println("Loss: ", state.loss)
```

For a complete, runnable example, see [`examples/pure_snn_training.jl`](examples/pure_snn_training.jl).

## Hybrid Usage (Teacher-Student Distillation)

`SynapticDistill.jl` can be used in hybrid systems where an SNN is trained to distill knowledge from a larger teacher model. Keep teacher-specific integration outside this package: capture the teacher targets in your loss function, and inject only the SNN's model step.

```julia
# Assume `teacher_targets` came from your application-specific teacher pipeline.
teacher_targets = get_teacher_targets(batch_id)

function model_step(model, spikes::SpikeBatch)
    return (logits = run_snn(model, spikes.spikes),)
end

loss_fn = output -> cross_entropy(output.logits, teacher_targets)

model, state = train_step!(model, spike_batch, loss_fn; forward_fn=model_step, rule=:eprop)
```

For a complete, runnable example, see [`examples/hybrid_moe_training.jl`](examples/hybrid_moe_training.jl).

## Available Rules

- `:eprop`: Eligibility propagation. **Stub** — currently only logs the rule name; `train_step!` does not apply the eligibility-trace update.
- `:ottt`: Online Spatio-Temporal Trace Training. **Stub** — currently only logs the rule name; `train_step!` does not apply the OTTT update.

## Custom Model Steps and Loss Functions

A `ModelStep` is any callable object with signature `(model, spikes::SpikeBatch) -> output`. The output can be any Julia value accepted by your loss function, such as a named tuple containing logits, spike counts, membrane potentials, or task-specific metrics.

Any function that takes that output and returns a scalar loss is a valid loss function.

## Spikenaut sidecar (`scripts/spikenaut_train.jl`)

Standalone trainer (JSON3 + stdlib only — it does **not** `using SynapticDistill`). This is the path that writes the 16×16 LIF `snn_model.json` and signed Q8.8 `.mem` files. Outgoing Dale (readout only) and K-WTA stay on during training; incoming `W` has no E/I sign. Health evaluation is `k=none` on **test** `gpu-000170..198` (mean pairwise cofire, all-16, I spikes) and does not mutate the bank that `export_artifacts` serializes. CLI split must be **train** (default); `val` / `test` error instead of `tick!(learn=true)` on the holdout. A JSONL with no test episodes errors instead of silently evaluating train. JSON `null` on a live key still counts; the value encodes as 0 (T=0 stays 0).

It trains on the **legal v3 `state_telemetry` encoder**, not `qubic_ticks_snn` `*_derived` columns (those are a closed form of `tick_rate`; Spikenaut Scientist exp-008, 0 mismatches / 27430). Pointing the sidecar at derived-only JSONL errors instead of silently training on forbidden sensors.

**Live columns** (axons 0..4; survive train AND val AND test; exp-008):

`mem_util_pct`, `power_w`, `gpu_temp_c`, `sm_clock_mhz`, `mem_clock_mhz`

Axons 5..15 are unused width, held at 0 — not fake channels and not first-differences. Do not use `fan_speed_pct` or `vddcr_gfx_v` (constant on val), `vram_temp_c` (`gpu_temp+8` except idle 0), or `step_idx`. Reward and readout targets use live `power_w` and `gpu_temp_c`.

**Frozen minmax** (v3 `state_telemetry` train split, sha lineage `74acdd0f`). Clamp to `[0, 1]` after scale. Do not refit on val/test:

| column | min | max |
| --- | --- | --- |
| `mem_util_pct` | 0 | 75 |
| `power_w` | 8.527000427246094 | 302.8450012207031 |
| `gpu_temp_c` | 0 | 69 |
| `sm_clock_mhz` | 180 | 2910 |
| `mem_clock_mhz` | 405 | 14801 |

**Episode holdout** (`episode_id`; never shuffle rows across episodes; `ts_utc` is not invented):

- train `gpu-000000..138`
- embargo 139
- val `gpu-000140..168`
- embargo 169
- test `gpu-000170..198`

Ingest is JSONL. Published v3 shards are parquet; this sidecar does not convert them or fabricate timestamps. Pass JSONL records that already carry the five live fields plus `episode_id`.

The sidecar has its own environment (`scripts/Project.toml`) because JSON3 is
not a dependency of the package itself — run it with `--project=scripts`, not
`--project=.`:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once

julia --project=scripts scripts/spikenaut_train.jl \
  /path/to/state_telemetry.jsonl \
  20 /tmp/spikenaut-out train
```

Library `update_eprop!` / `update_ottt!` stay stubs; do not add this package to the Rust `Cargo.toml`. Do not export weights to Hugging Face and do not write `rmems/Spikenaut-SNN` `dataset/merged_v2/` from this path.

Cite: **Spikenaut Scientist** · exp-008..014.

## Integration

`SynapticDistill.jl` is intentionally framework-agnostic. Application-specific IPC, teacher-model execution, hardware interfaces, and domain-specific model architectures should live in caller code and connect through injected model-step and loss callbacks.

## License

This project is dual-licensed under either the [MIT License](LICENSE) or the [Apache License 2.0](LICENSE-APACHE), at your option.

## Contributing

Contributions are welcome! Please open an issue or pull request to discuss your ideas.
