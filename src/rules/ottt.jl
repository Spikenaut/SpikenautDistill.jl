# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Online Spatio-Temporal Trace Training (OTTT).

Unlike e-prop’s time-averaged eligibility, OTTT accumulates the outer
product of the learning signal and the *per-timestep* presynaptic trace:

`ΔW += (1/T) ∑_t L ⊗ y[t]`, `y[t] = λ y[t-1] + pre[t]`.
"""

"""
    update_ottt!(model, spikes, loss, output; traces, trace_lambda, loss_fn, kwargs...)

Return `(gradients, traces)` matching [`update_eprop!`](@ref).
"""
function update_ottt!(model, spikes::SpikeBatch, loss, output;
                      traces = nothing,
                      trace_lambda::Real = 0.95f0,
                      loss_fn = nothing,
                      kwargs...)
    W = _weights(model)
    n_out, n_pre = size(W)
    S = _spike_matrix(spikes, n_pre)
    λ = Float32(trace_lambda)
    T = size(S, 2)

    carry = nothing
    if traces isa TraceBatch && traces.traces isa NamedTuple && haskey(traces.traces, :pre)
        carry = traces.traces.pre
    end

    y, _, per_t = _presynaptic_traces(S, λ, carry)
    L = loss_fn === nothing ? ones(Float32, n_out) : _learning_signal(loss_fn, output, n_out)

    grads = zeros(Float32, n_out, n_pre)
    if T > 0
        @inbounds for t in 1:T
            @views grads .+= L * per_t[:, t]'
        end
        grads ./= T
    end

    new_traces = TraceBatch((pre = y, eligibility = grads, rule = :ottt))
    return grads, new_traces
end
