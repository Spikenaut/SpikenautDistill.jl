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
    _spikes_as_matrix(s) -> AbstractMatrix

Normalize the `SpikeBatch.spikes` representations documented in `types.jl` — an
`AbstractMatrix` (sparse included) or a vector of equal-length vectors — to a
matrix. Each inner vector becomes one column; `_spike_matrix` then settles the
channel/time orientation.
"""
_spikes_as_matrix(s::AbstractMatrix) = s
function _spikes_as_matrix(s::AbstractVector)
    isempty(s) && throw(ArgumentError("SpikeBatch.spikes is empty"))
    all(v -> v isa AbstractVector, s) || throw(ArgumentError(
        "SpikeBatch.spikes must be an AbstractMatrix (channels × time) or a vector " *
        "of equal-length vectors, got a vector of $(eltype(s))"))
    n = length(first(s))
    all(v -> length(v) == n, s) || throw(ArgumentError(
        "SpikeBatch.spikes vector-of-vectors needs equal-length inner vectors, got " *
        "lengths $(sort!(unique(map(length, s))))"))
    # `reduce(hcat, ·)` on a length-1 `Vector{Any}` returns the inner vector.
    return hcat(s...)
end
_spikes_as_matrix(s) = throw(ArgumentError(
    "SpikeBatch.spikes must be an AbstractMatrix (channels × time) or a vector of " *
    "equal-length vectors, got $(typeof(s))"))

"""
    _spike_matrix(batch, n_pre) -> AbstractMatrix

`SpikeBatch.spikes` as `n_pre × T`. Rows are presynaptic channels matching
`size(model.weights, 2)`. A `T × n_pre` layout is transposed. Accepts both
representations `types.jl` documents — see [`_spikes_as_matrix`](@ref).

A square `n_pre × n_pre` batch is **rejected**: channel-major and time-major are
indistinguishable at that shape, and guessing silently transposes the eligibility
axes and corrupts the gradient direction. Reshape to an unambiguous `T`, or pass
the canonical orientation explicitly.
"""
function _spike_matrix(batch::SpikeBatch, n_pre::Integer)
    s = _spikes_as_matrix(batch.spikes)
    nr, nc = size(s)
    if nr == n_pre && nc == n_pre
        throw(ArgumentError(
            "SpikeBatch.spikes is square ($nr × $nc) with n_pre=$n_pre, so the " *
            "channel/time orientation is ambiguous and picking one would silently " *
            "transpose the traces. Pass a batch whose T differs from n_pre."))
    elseif nr == n_pre
        return s
    elseif nc == n_pre
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
    _logits_rebuilder(output) -> (z -> output′)

Closure replacing `output.logits` with `z` while keeping every other property.

`NamedTuple`s round-trip exactly. Any other type is rebuilt as a `NamedTuple` of
its properties: the fields a `loss_fn` reads survive, but the concrete type does
not, so a `loss_fn` dispatching on that type raises — and `_learning_signal`
surfaces that rather than quietly optimizing a different objective.
"""
function _logits_rebuilder(output)
    output isa NamedTuple && return z -> merge(output, (logits = z,))
    props = propertynames(output)
    base = NamedTuple{props}(map(p -> getproperty(output, p), props))
    return z -> merge(base, (logits = z,))
end

"""
    _learning_signal(loss_fn, output, n_out) -> Vector{Float32}

`∂L/∂logits` when `output` has `.logits`. Any other properties of `output` are
carried through, so a `loss_fn` that reads them still works — see
[`_logits_rebuilder`](@ref) for the one case that cannot be preserved.

Throws when `output` has `.logits` but the derivative is unavailable or the
wrong length. `:eprop` / `:ottt` use this vector *as* the learning signal — the
Zygote path in `train_step!` only feeds `:surrogate` — so falling back to a
uniform signal there would apply an unsigned update unrelated to `loss`. The
uniform signal is used only when `output` carries no `.logits` at all.
"""
function _learning_signal(loss_fn, output, n_out::Integer)
    (output === nothing || !hasproperty(output, :logits)) && return ones(Float32, n_out)

    z0 = _as_f32_vec(output.logits)
    rebuild = _logits_rebuilder(output)
    g = try
        Zygote.gradient(z -> loss_fn(rebuild(z)), z0)[1]
    catch err
        throw(ArgumentError(
            "could not differentiate `loss_fn` w.r.t. `output.logits`: " *
            "$(sprint(showerror, err)); e-prop / OTTT need `∂L/∂logits`"))
    end
    g === nothing && throw(ArgumentError(
        "`loss_fn` has no derivative w.r.t. `output.logits`; " *
        "e-prop / OTTT need a non-`nothing` `∂L/∂logits`"))
    L = _as_f32_vec(g)
    length(L) == n_out || throw(DimensionMismatch(
        "`∂L/∂logits` length $(length(L)) != n_out=$n_out"))
    return L
end

"""
    _learning_signal_per_t(loss_fn, output, n_out, T) -> (Matrix{Float32}, Bool)

Per-timestep `∂L/∂logits` as `n_out × T`, plus whether it is genuinely
time-resolved.

This is what separates OTTT from e-prop. When `output.logits` is an `n_out × T`
matrix, the gradient is taken through the whole matrix, so column `t` is the
learning signal *at* `t` and `∑ₜ L[:,t] ⊗ y[t]` cannot be refactored into
`L ⊗ ȳ`. When `logits` is a single vector there is no per-timestep information
to recover: the episode-level signal is broadcast across `T`, and OTTT
necessarily collapses back onto e-prop. The returned `Bool` says which happened,
so callers can tell a real per-timestep rule from the degenerate case.
"""
function _learning_signal_per_t(loss_fn, output, n_out::Integer, T::Integer)
    if output !== nothing && hasproperty(output, :logits) && output.logits isa AbstractMatrix
        Z = Float32.(output.logits)
        size(Z, 1) == n_out || throw(DimensionMismatch(
            "`output.logits` has $(size(Z, 1)) rows, expected n_out=$n_out"))
        size(Z, 2) == T || throw(DimensionMismatch(
            "per-timestep `output.logits` has $(size(Z, 2)) columns, expected T=$T"))
        rebuild = _logits_rebuilder(output)
        G = try
            Zygote.gradient(z -> loss_fn(rebuild(z)), Z)[1]
        catch err
            throw(ArgumentError(
                "could not differentiate `loss_fn` w.r.t. per-timestep " *
                "`output.logits`: $(sprint(showerror, err))"))
        end
        G === nothing && throw(ArgumentError(
            "`loss_fn` has no derivative w.r.t. per-timestep `output.logits`"))
        return Float32.(G), true
    end
    L = _learning_signal(loss_fn, output, n_out)
    return repeat(L, 1, max(T, 0)), false
end

"""
    _presynaptic_traces(S, λ, carry) -> (final_trace, time_mean, per_t)

Leaky pre-trace `y[t] = λ y[t-1] + S[:, t]` over an `n_pre × T` spike matrix.

`λ` must be finite and in `[0, 1)`: the `1 - λ` rate scaling below makes every
gradient identically zero at `λ == 1`, and negative — i.e. gradient *ascent* —
above it.
"""
function _presynaptic_traces(S::AbstractMatrix, λ::Float32, carry::Union{Nothing,AbstractVector})
    isfinite(λ) && 0f0 <= λ < 1f0 || throw(ArgumentError(
        "trace_lambda must be finite and in [0, 1), got $λ; " *
        "λ = 1 zeroes every gradient and λ > 1 inverts the update direction"))
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
