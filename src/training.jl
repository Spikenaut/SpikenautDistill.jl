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

# A placeholder for a default optimizer.
function default_optimizer()
    # In a real scenario, this would return a configured optimizer from a library like Optimisers.jl.
    return (params, grads) -> params .-= 0.001f0 .* grads
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

    # Accept Function, ModelStep subtypes, or any object that has methods (callable).
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

Perform one online training step using the chosen rule.

- `model`: Your SNN model.
- `spikes`: A `SpikeBatch` containing spike trains and optional targets.
- `loss_fn`: A function that takes the model output and computes a scalar loss.
- `forward_fn` / `model_step`: A caller-provided callable with signature
  `(model, spikes::SpikeBatch) -> output`.

Returns: A tuple of `(updated_model, TrainingState)`.
"""
function train_step!(model, spikes::SpikeBatch, loss_fn;
                     forward_fn = nothing,
                     rule::Symbol = :eprop,
                     optimizer = default_optimizer(),
                     kwargs...)

    model_step = validate_model_step(forward_fn)

    # Differentiate through the injected model step so gradients depend on model params.
    # Zygote.withgradient returns a NamedTuple `(val, grad)`.
    result = Zygote.withgradient(model) do m
        output = model_step(m, spikes)
        loss_fn(output)
    end
    loss = result.val
    # withgradient(model) returns grad as a 1-tuple (one entry per AD argument).
    g = result.grad
    grads = g === nothing ? nothing : (g isa Tuple ? g[1] : g)

    loss isa Number || throw(ArgumentError(
        "`loss_fn` must return a numeric scalar, got $(typeof(loss))."))
    loss = Float32(loss)

    # Apply the chosen learning rule.
    if rule == :eprop
        # The `update_eprop!` function will calculate eligibility traces and gradients.
        # update_eprop!(model, spikes, loss, output; kwargs...)
        println("Applying e-prop rule (not fully implemented).")
    elseif rule == :ottt
        # update_ottt!(model, spikes, loss; kwargs...)
        println("Applying OTTT rule (not fully implemented).")
    else
        error("Unknown training rule: `$rule`")
    end

    # Apply gradients (this is a simplified view).
    # In a real implementation, the rule-specific function would return gradients
    # to be applied here.
    # optimizer(... , grads)

    # Return the updated model and training state.
    state = TrainingState(loss=loss, gradients=grads)
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
