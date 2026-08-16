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

- `:eprop`: Eligibility propagation.
- `:ottt`: Online Spatio-Temporal Trace Training.

## Custom Model Steps and Loss Functions

A `ModelStep` is any callable object with signature `(model, spikes::SpikeBatch) -> output`. The output can be any Julia value accepted by your loss function, such as a named tuple containing logits, spike counts, membrane potentials, or task-specific metrics.

Any function that takes that output and returns a scalar loss is a valid loss function.

## Spikenaut sidecar (`scripts/spikenaut_train.jl`)

Standalone trainer (JSON3 + stdlib only — it does **not** `using SynapticDistill`). This is the path that writes the 16×16 LIF `snn_model.json` and signed Q8.8 `.mem` files consumed by [rmems/Spikenaut-SNN](https://github.com/rmems/Spikenaut-SNN) `dataset/merged_v2/`.

The sidecar has its own environment (`scripts/Project.toml`) because JSON3 is
not a dependency of the package itself — run it with `--project=scripts`, not
`--project=.`:

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'   # once

julia --project=scripts scripts/spikenaut_train.jl \
  /home/raulmc/Spikenaut-Vault/Spikenaut-SNN-Telemetry/full_data/qubic_ticks_snn.jsonl \
  20 /tmp/spikenaut-out
```

Train on `qubic_ticks_snn.jsonl` (~27k rows). The 8-record `fresh_sync` sample produces the monotonic all-positive hidden-weight artifact. Library `update_eprop!` / `update_ottt!` stay stubs; do not add this package to the Rust `Cargo.toml`. Hugging Face `rmems/Spikenaut-SNN` is a weight mirror only — commit artifacts on GitHub first.

## Integration

`SynapticDistill.jl` is intentionally framework-agnostic. Application-specific IPC, teacher-model execution, hardware interfaces, and domain-specific model architectures should live in caller code and connect through injected model-step and loss callbacks.

## License

This project is dual-licensed under either the [MIT License](LICENSE) or the [Apache License 2.0](LICENSE-APACHE), at your option.

## Contributing

Contributions are welcome! Please open an issue or pull request to discuss your ideas.
