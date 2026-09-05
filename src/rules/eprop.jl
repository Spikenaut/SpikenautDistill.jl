# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Eligibility-propagation (e-prop) for a linear readout on spike trains.

`e_j[t] = λ e_j[t-1] + pre_j[t]`, then `∂L/∂W_ij ≈ L_i * ē_j`.
`L = ∂L/∂logits` from the caller loss. No membrane state is required; the
surrogate lives in the injected model step / Zygote path when the caller
uses one.
"""

"""
    update_eprop!(model, spikes, loss, output; traces, trace_lambda, loss_fn, kwargs...)

Return `(gradients, traces)` where `gradients` is `n_out × n_pre` matching
`model.weights`. `traces` is a [`TraceBatch`](@ref) carrying the pre-trace
so the next tick can continue the eligibility filter.
"""
function update_eprop!(model, spikes::SpikeBatch, loss, output;
                       traces = nothing,
                       trace_lambda::Real = 0.95f0,
                       loss_fn = nothing,
                       kwargs...)
    W = _weights(model)
    n_out, n_pre = size(W)
    S = _spike_matrix(spikes, n_pre)
    λ = Float32(trace_lambda)

    carry = nothing
    if traces isa TraceBatch && traces.traces isa NamedTuple && haskey(traces.traces, :pre)
        carry = traces.traces.pre
    end

    y, mean_y, _ = _presynaptic_traces(S, λ, carry)
    L = loss_fn === nothing ? ones(Float32, n_out) : _learning_signal(loss_fn, output, n_out)
    grads = L * mean_y'                          # n_out × n_pre
    new_traces = TraceBatch((pre = y, eligibility = grads, rule = :eprop))
    return grads, new_traces
end
