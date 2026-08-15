# SPDX-License-Identifier: MIT OR Apache-2.0

"""Shared helpers for online rules and `train_step!`."""

function _weights(model)
    hasproperty(model, :weights) || throw(ArgumentError(
        "model must have a `.weights` field (Matrix) for e-prop / OTTT"))
    return model.weights
end

function _weight_grad(grads)
    grads === nothing && return nothing
    grads isa AbstractArray && return grads
    if grads isa NamedTuple && haskey(grads, :weights)
        return grads.weights
    end
    hasproperty(grads, :weights) && return getproperty(grads, :weights)
    return nothing
end

"""
    _spike_matrix(batch, n_pre) -> AbstractMatrix

`SpikeBatch.spikes` as `n_pre × T`. Rows are presynaptic channels matching
`size(model.weights, 2)`. A `T × n_pre` layout is transposed.
"""
function _spike_matrix(batch::SpikeBatch, n_pre::Integer)
    s = batch.spikes
    s isa AbstractMatrix || throw(ArgumentError(
        "SpikeBatch.spikes must be an AbstractMatrix (channels × time), got $(typeof(s))"))
    if size(s, 1) == n_pre
        return s
    elseif size(s, 2) == n_pre
        return permutedims(s)
    else
        throw(ArgumentError(
            "SpikeBatch.spikes size $(size(s)) is incompatible with n_pre=$n_pre"))
    end
end

function _as_f32_vec(x)
    x isa AbstractVector && return Float32.(x)
    x isa AbstractArray && return Float32.(vec(x))
    return Float32[Float32(x)]
end

"""
    _learning_signal(loss_fn, output, n_out) -> Vector{Float32}

`∂L/∂logits` when `output` has `.logits`; otherwise a uniform signal of
length `n_out` (still signed by `loss` via the caller’s Zygote path).
"""
function _learning_signal(loss_fn, output, n_out::Integer)
    if output !== nothing && hasproperty(output, :logits)
        z0 = Float32.(_as_f32_vec(output.logits))
        g = try
            Zygote.gradient(z -> loss_fn((logits = z,)), z0)[1]
        catch
            nothing
        end
        if g !== nothing
            L = _as_f32_vec(g)
            length(L) == n_out && return L
        end
    end
    return ones(Float32, n_out)
end

"""
    _presynaptic_traces(S, λ, carry) -> (final_trace, time_mean, per_t)

Leaky pre-trace `y[t] = λ y[t-1] + S[:, t]` over an `n_pre × T` spike matrix.
"""
function _presynaptic_traces(S::AbstractMatrix, λ::Float32, carry::Union{Nothing,AbstractVector})
    n_pre, T = size(S)
    y = carry === nothing ? zeros(Float32, n_pre) : Float32.(copy(carry))
    length(y) == n_pre || throw(ArgumentError(
        "carried pre-trace length $(length(y)) != n_pre=$n_pre"))
    acc = zeros(Float32, n_pre)
    per_t = Matrix{Float32}(undef, n_pre, T)
    @inbounds for t in 1:T
        @views y .= λ .* y .+ Float32.(S[:, t])
        per_t[:, t] .= y
        acc .+= y
    end
    mean_y = T == 0 ? acc : acc ./ T
    # (1-λ) keeps ē on a firing-rate scale so SGD does not explode.
    # Carry `y` stays unscaled so the next tick continues the same filter.
    scale = 1f0 - λ
    return y, scale .* mean_y, scale .* per_t
end

function apply_weight_update!(model, grads, optimizer)
    gW = _weight_grad(grads)
    (gW === nothing || !hasproperty(model, :weights)) && return model
    W = model.weights
    G = reshape(Float32.(gW), size(W))
    optimizer(W, G)
    return model
end
