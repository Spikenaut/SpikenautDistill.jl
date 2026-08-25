# SPDX-License-Identifier: MIT OR Apache-2.0

using SynapticDistill
using Statistics

# This example simulates a hybrid training scenario where the loss is derived
# from an external teacher model.

# 1. Define a simple SNN model.
mutable struct SimpleSNN
    weights::Matrix{Float32}
end

# 2. Dummy data representing spikes from the SNN and targets from a teacher.
spike_data = Float32.(rand(0:1, 10, 100))
teacher_targets = rand(Float32, 10)
spike_batch = SpikeBatch(spike_data, nothing, teacher_targets)

# 3. Define a model instance.
model = SimpleSNN(rand(Float32, 10, 10))

# 4. Define an injected model step and loss function that uses the teacher targets.
function model_step(model, spikes::SpikeBatch)
    rates = vec(mean(spikes.spikes; dims=2))
    return (logits = model.weights * rates,)
end

# The key idea is that `loss_fn` is a closure that captures the targets.
function cross_entropy_loss(output, targets)
    # A real implementation would compute cross-entropy between the SNN's output logits
    # and the target distribution from the teacher.
    return sum((output.logits .- targets) .^ 2)
end

# Create a closure that captures the teacher targets.
loss_fn = (output) -> cross_entropy_loss(output, teacher_targets)

# 5. Run a training step.
println("Running a single hybrid training step...")
model, state = train_step!(model, spike_batch, loss_fn; forward_fn=model_step, rule=:eprop)

println("Training step complete.")
println("Loss: ", state.loss)

# In a real hybrid setup, caller code would decide how to consume `state.gradients`.
# println("Gradients: ", state.gradients)
