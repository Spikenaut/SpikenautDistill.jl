# SPDX-License-Identifier: MIT OR Apache-2.0

# Hybrid teacher → student: the teacher stays outside this package.
# Caller supplies frozen targets; only the SNN student is updated.

using SynapticDistill
using Statistics

mutable struct SimpleSNN
    weights::Matrix{Float32}
end

spike_data = Float32.(rand(0:1, 10, 100))
teacher_targets = rand(Float32, 10)
spike_batch = SpikeBatch(spike_data, nothing, teacher_targets)

model = SimpleSNN(rand(Float32, 10, 10))

function model_step(model, spikes::SpikeBatch)
    rates = vec(mean(spikes.spikes; dims=2))
    return (logits = model.weights * rates,)
end

# MSE stand-in for a distillation loss against precomputed teacher targets.
loss_fn = output -> sum(abs2, output.logits .- teacher_targets)

println("Running a hybrid teacher→student step...")
model, state = train_step!(model, spike_batch, loss_fn; forward_fn=model_step, rule=:eprop)
println("Loss: ", state.loss)
