# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Main training entry point for SynapticDistill.
"""

"""
    ModelStep

Callable interface for model forward steps used by `train_step!`.

A model step is any callable with the signature
`model_step(model, spikes::SpikeBatch) -> output`. The returned `output` is
passed unchanged to the caller-provided loss function.

Subtypes may implement the call overload:

```julia
struct MyStep <: ModelStep end
(::MyStep)(model, spikes::SpikeBatch) = ...
```

Plain `Function`s and other callable objects are also accepted by `train_step!`.
"""
abstract type ModelStep end

function default_optimizer(lr::Real = 0.001f0)
    lr32 = Float32(lr)
    return (params, grads) -> params .-= lr32 .* grads
end

"""
    validate_model_step(model_step)

Validate that `model_step` is present and callable.

Accepts:
- plain `Function`s
- subtypes of [`ModelStep`](@ref)
- any other object with methods (callable structs)

Does not fully pre-check arity/signature: `hasmethod(..., Tuple{Any,SpikeBatch})`
rejects callables typed to a concrete model type, and applicability needs a real
model instance. Incompatible callables therefore surface as `MethodError` at the
call site inside `train_step!`.
"""
function validate_model_step(model_step)
    model_step === nothing && throw(ArgumentError(
        "`forward_fn` is required; pass a callable `(model, spikes::SpikeBatch) -> output` to `train_step!`."))

    if model_step isa Function || model_step isa ModelStep
        return model_step
    end

    local has_methods = false
    try
        has_methods = !isempty(methods(model_step))
    catch
        has_methods = false
    end
    has_methods || throw(ArgumentError(
        "`forward_fn` must be callable as `(model, spikes::SpikeBatch) -> output`."))
    return model_step
end

"""
    train_step!(model, spikes::SpikeBatch, loss_fn; forward_fn, rule=:eprop, kwargs...)
    train_step!(model, spikes::SpikeBatch, loss_fn, model_step; rule=:eprop, kwargs...)

One online step. Computes the scalar loss through the injected model step,
builds e-prop or OTTT gradients from the spike train, and **applies** them
to `model.weights`.

Keyword arguments:
- `forward_fn` / positional `model_step`: `(model, spikes) -> output`
- `rule`: `:eprop` or `:ottt`
- `optimizer`: `(params, grads) -> ...` (default SGD, `lr=0.001`)
- `traces`: optional [`TraceBatch`](@ref) carried from the previous step
- `trace_lambda`: eligibility / OTTT leak (default `0.95`)

Returns `(updated_model, TrainingState)` with `state.gradients` and
`state.traces` set.
"""
function train_step!(model, spikes::SpikeBatch, loss_fn;
                     forward_fn = nothing,
                     rule::Symbol = :eprop,
                     optimizer = default_optimizer(),
                     traces = nothing,
                     kwargs...)

    model_step = validate_model_step(forward_fn)
    output_ref = Ref{Any}(nothing)

    result = Zygote.withgradient(model) do m
        output = model_step(m, spikes)
        Zygote.ignore_derivatives() do
            output_ref[] = output
        end
        loss_fn(output)
    end
    loss = result.val
    g = result.grad
    zygote_grads = g === nothing ? nothing : (g isa Tuple ? g[1] : g)

    loss isa Number || throw(ArgumentError(
        "`loss_fn` must return a numeric scalar, got $(typeof(loss))."))
    loss = Float32(loss)
    output = output_ref[]

    rule_grads, new_traces = if rule === :eprop
        update_eprop!(model, spikes, loss, output;
                      traces = traces, loss_fn = loss_fn, kwargs...)
    elseif rule === :ottt
        update_ottt!(model, spikes, loss, output;
                     traces = traces, loss_fn = loss_fn, kwargs...)
    elseif rule === :surrogate
        (zygote_grads, traces)
    else
        error("Unknown training rule: `$rule`")
    end

    apply_weight_update!(model, rule_grads, optimizer)

    state = TrainingState(loss = loss, traces = new_traces, gradients = rule_grads)
    return model, state
end

function train_step!(model, spikes::SpikeBatch, loss_fn, model_step;
                     rule::Symbol = :eprop,
                     optimizer = default_optimizer(),
                     kwargs...)
    return train_step!(model, spikes, loss_fn;
                       forward_fn = model_step,
                       rule = rule,
                       optimizer = optimizer,
                       kwargs...)
end
