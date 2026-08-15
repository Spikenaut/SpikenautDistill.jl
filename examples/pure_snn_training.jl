# SPDX-License-Identifier: MIT OR Apache-2.0

using SynapticDistill
using LinearAlgebra
using Statistics

# 1. Define a simple SNN model.
# In a real scenario, this would be a more complex model from a library like Flux or a custom struct.
mutable struct SimpleSNN
    weights::Matrix{Float32}
end

# 2. Create some dummy data.
# Spike train (e.g., 10 neurons, 100 timesteps)
spike_data = Float32.(rand(0:1, 10, 100))
spike_batch = SpikeBatch(spike_data, nothing, rand(10))

# 3. Define a model instance.
model = SimpleSNN(rand(Float32, 10, 10))

# 4. Define an injected model step and loss function.
function model_step(model, spikes::SpikeBatch)
    rates = vec(mean(spikes.spikes; dims=2))
    return (logits = model.weights * rates,)
end

function mse_loss(output)
    return sum(output.logits .^ 2)
end

println("Running a single training step...")
model, state = train_step!(model, spike_batch, mse_loss; forward_fn=model_step, rule=:eprop)

println("Training step complete.")
println("Loss: ", state.loss)
println("‖∇W‖: ", norm(state.gradients))
