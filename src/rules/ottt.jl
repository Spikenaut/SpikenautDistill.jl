# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Online Spatio-Temporal Trace Training (OTTT).

Unlike e-prop’s time-averaged eligibility, OTTT pairs a *per-timestep* learning
signal with the *per-timestep* presynaptic trace:

`ΔW += (1/T) ∑_t L[t] ⊗ y[t]`, `y[t] = λ y[t-1] + pre[t]`.

The per-timestep `L[t]` is the load-bearing part. With a single episode-level
`L`, `∑ₜ L ⊗ y[t]` factors into `L ⊗ ∑ₜ y[t]` — exactly e-prop's `L ⊗ ȳ` — and
the two rules produce identical gradients. To get a genuinely distinct rule the
model step must emit `output.logits` as an `n_out × T` matrix; `update_ottt!`
then differentiates through it per column. A vector `logits` still works but
degenerates to e-prop, and `traces.time_resolved` reports which case ran.
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
    Lt, time_resolved = loss_fn === nothing ?
        (ones(Float32, n_out, T), false) :
        _learning_signal_per_t(loss_fn, output, n_out, T)

    grads = zeros(Float32, n_out, n_pre)
    if T > 0
        @inbounds for t in 1:T
            @views grads .+= Lt[:, t] * per_t[:, t]'
        end
        grads ./= T
    end

    new_traces = TraceBatch(
        (pre = y, eligibility = grads, rule = :ottt, time_resolved = time_resolved))
    return grads, new_traces
end
