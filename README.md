# SynapticDistill.jl

[![CI](https://github.com/rmems/SynapticDistill.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/rmems/SynapticDistill.jl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

**Repository:** [`rmems/SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)

**Modular online training for spiking neural networks in Julia — E-prop, OTTT, and more. Works with pure SNNs or hybrid teacher-student systems.**

`SynapticDistill.jl` is a flexible and performant library for training spiking neural networks (SNNs) using online (event-based) learning rules. It is designed to be framework-agnostic, allowing researchers to bring their own models, model-step callbacks, loss functions, and data sources. Canonical home is [`rmems/SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl) after the Limen-Neural → rmems transfer (`Limen-Neural/SynapticDistill.jl` redirects here).

```julia
using Pkg
Pkg.add(url="https://github.com/rmems/SynapticDistill.jl")
```

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

This package lives under **rmems**. The STDP / Hebbian peer has **not** transferred (`rmems/plasticity-lab` does not exist). Keep linking the still-live external peer; do not invent an rmems plasticity-lab URL.

- `SynapticDistill.jl` (Julia, [`rmems/SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl)): differentiable or online distillation and teacher-student knowledge transfer.
- [`plasticity-lab`](https://github.com/Limen-Neural/plasticity-lab) (Rust; still `Limen-Neural/plasticity-lab`): reward-modulated STDP / Hebbian plasticity rules and online low-level weight delta computation.
- `SynapticDistill.jl` must not become the home for STDP logic; `plasticity-lab` must not absorb distillation logic.

See the matching boundary note in the [`plasticity-lab` README](https://github.com/Limen-Neural/plasticity-lab#scope-and-ownership-boundaries) (intentional external-peer link) and the Linear issue [LIM-25](https://linear.app/rpd-34/issue/LIM-25/plasticity-lab-clarify-ownership-boundary-with-synapticdistilljl).

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

Standalone trainer (JSON3 + stdlib only — it does **not** `using SynapticDistill`). When pointed at legal v3 `state_telemetry` JSONL, this script **can** train and export signed two's-complement Q8.8, outgoing Dale 12:4 (75:25; 4 I — informal “80:20” was Dale shorthand for the I count, not the ratio), and K-WTA. That is a **capability** of the sidecar, not a claim that `rmems/Spikenaut-SNN` `dataset/merged_v2/` has already been replaced, that Hamming / Spikenaut-SNN#4 is cleared, that Spikenaut-SNN#13 is closed, that Hugging Face was published, or that silicon weight-load is ready.

A run writes a **local** 16×16 LIF `snn_model.json` and signed Q8.8 `.mem` files into the `out_dir` you pass. Outgoing Dale (readout only) and K-WTA stay on during training; incoming `W` is **signed-capable** (`W_MIN < 0`) and has no Dale sign lock — "unsigned" would mean the old smoking-gun Q8.8 clamp, which this script does not use. Health evaluation is `k=none` on **test** `gpu-000170..198` (mean pairwise cofire, all-16, I spikes) and does not mutate the bank that `export_artifacts` serializes. CLI split must be **train** (default); `val` / `test` error instead of `tick!(learn=true)` on the holdout. A JSONL with no test episodes errors instead of silently evaluating train. JSON `null` on a live key still counts; the value encodes as 0 (T=0 stays 0).

**Anti-clone knobs** (Scientist exp-023 measured these on a 5-ep local run, seed 123: cofire 0.733, I live, 10/12 unique active Q8.8). The knobs are in this script: live-row cosine repulsion (`DIV_LR=0.00035`, `DIV_COS_MIN=0.55`), I-drive (`I_DRIVE=0.05`, `I_THRESH=0.90`), train K-WTA quota (`I_WTA_MAX=2`, `E_WTA_MIN=2`), milder LTD (`STDP_LTD=0.0008`), homeostatic thresh (`RATE_TARGET=0.12`). Optional CLI seed (default 123). Encoder / holdout / health_eval contract unchanged. That measurement is **not** a promoted `merged_v2` health-PASS.

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

Ingest is JSONL. Published v3 shards are parquet; this sidecar does not convert them or fabricate timestamps. Pass JSONL records that already carry the five live fields plus a six-digit `episode_id` (`gpu-######`). A missing or malformed v3 `episode_id` errors instead of silently dropping the row.

The sidecar has its own environment (`scripts/Project.toml`) because JSON3 is
not a dependency of the package itself — run it with `--project=scripts`, not
`--project=.`:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once

julia --project=scripts scripts/spikenaut_train.jl \
  /path/to/state_telemetry.jsonl \
  5 /tmp/spikenaut-out train 123
```

Library `update_eprop!` / `update_ottt!` stay stubs; do not add this package to the Rust `Cargo.toml`.

**`merged_v2` replacement source (capability, not done).** This script can write the five-file FPGA layout below to a local `out_dir`. A later health-PASS retrain *could* replace `rmems/Spikenaut-SNN` `dataset/merged_v2/` from those files. **None of that has happened from this path:** the sidecar does not write the Spikenaut-SNN tree, does not publish Hugging Face, does not load silicon, and does not overwrite `merged_v2`. Spikenaut-SNN#13 stays open. Promotion would be a separate Spikenaut-SNN PR after k=none test bars (exp-009 / exp-023). Do not treat the published ramp as this script's output.

| file | count / role |
| --- | --- |
| `snn_model.json` | 16 neurons + encoder / Dale / signed-Q8.8 metadata |
| `parameters.mem` | 16 thresholds |
| `parameters_weights.mem` | 256 signed Q8.8 hidden (neuron-major) |
| `parameters_decay.mem` | 16 keep factors (`0.85`) |
| `parameters_output_weights.mem` | 48 signed Q8.8 readout (neuron-major, Dale 12:4) |

Cite: **Spikenaut Scientist** · exp-008..023.

## Integration

`SynapticDistill.jl` is intentionally framework-agnostic. Application-specific IPC, teacher-model execution, hardware interfaces, and domain-specific model architectures should live in caller code and connect through injected model-step and loss callbacks.

## Wiki

The GitHub wiki is **enabled under rmems** (`has_wiki: true`). Browse [`rmems/SynapticDistill.jl/wiki`](https://github.com/rmems/SynapticDistill.jl/wiki) (HTTP 200). Wiki remote: `https://github.com/rmems/SynapticDistill.jl.wiki.git`. Pages already exist (Home, Overview, Getting Started, and the rest of the transferred map). Live install / blob links on those pages should use `rmems/SynapticDistill.jl`. Generated Devin DeepWiki footers that still cite a pre-transfer `Limen-Neural/SynapticDistill.jl` snapshot are historical provenance only.

## License

This project is dual-licensed under either the [MIT License](LICENSE) or the [Apache License 2.0](LICENSE-APACHE), at your option.

## Contributing

Contributions are welcome! Please open an issue or pull request to discuss your ideas.
