# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using SynapticDistill
using LinearAlgebra
using Random
using Statistics
using Zygote

# Top-level mock model (structs cannot be defined inside @testset local scope).
mutable struct MockSNN
    weights::Matrix{Float32}
end

# Callable ModelStep subtype used to exercise the typed-callable path.
struct MockStep <: ModelStep end
function (::MockStep)(model::MockSNN, batch::SpikeBatch)
    rates = vec(mean(batch.spikes; dims=2))
    return (logits = model.weights * rates,)
end

@testset "SynapticDistill" begin

    @testset "Package loads" begin
        @test @isdefined(SynapticDistill)
        @test SynapticDistill isa Module
        @test isdefined(SynapticDistill, :SpikeBatch)
        @test isdefined(SynapticDistill, :TraceBatch)
        @test isdefined(SynapticDistill, :TrainingState)
        @test isdefined(SynapticDistill, :ModelStep)
        @test isdefined(SynapticDistill, :train_step!)
        @test isdefined(SynapticDistill, :update_eprop!)
        @test isdefined(SynapticDistill, :update_ottt!)
        @test isdefined(SynapticDistill, :surrogate_heaviside)
        @test isdefined(SynapticDistill, :surrogate_sigmoid)
        @test isdefined(SynapticDistill, :surrogate_exponential)
    end

    @testset "surrogate gradients" begin
        # heaviside surrogate: at threshold → γ (10.0 default)
        @test surrogate_heaviside(0.0f0) ≈ 10.0f0 atol=0.01f0
        for v in -2.0f0:0.5f0:4.0f0
            @test surrogate_heaviside(v) ≥ 0.0f0
        end

        # sigmoid surrogate: at threshold → 0.25
        @test surrogate_sigmoid(0.0f0, 1.0f0) ≈ 0.25f0 atol=0.01f0
        # Always non-negative
        for v in -2.0f0:0.5f0:4.0f0
            @test surrogate_sigmoid(v, 1.0f0) ≥ 0.0f0
        end

        # exponential surrogate: at threshold → 1.0 (α * exp(0) = α)
        @test surrogate_exponential(0.0f0, 1.0f0) ≈ 1.0f0 atol=0.01f0
        for v in -2.0f0:0.5f0:4.0f0
            @test surrogate_exponential(v, 1.0f0) ≥ 0.0f0
        end
    end

    @testset "train_step! model step injection" begin
        model = MockSNN(Float32[1 2; 3 4])
        spikes = SpikeBatch(Float32[1 0 1; 0 1 1], nothing, nothing)
        calls = Ref(0)

        function mock_step(model, batch::SpikeBatch)
            # Side-effect must not be traced by Zygote (model step runs inside withgradient).
            Zygote.ignore_derivatives() do
                calls[] += 1
            end
            rates = vec(mean(batch.spikes; dims=2))
            return (logits = model.weights * rates,)
        end

        loss_fn(output) = sum(output.logits)

        # rates = [2/3, 2/3]; logits = W * rates = [2, 14/3]; sum = 20/3
        expected_loss = 20.0f0 / 3.0f0

        updated_model, state = train_step!(model, spikes, loss_fn; forward_fn=mock_step, rule=:eprop)

        @test updated_model === model
        @test calls[] == 1
        @test state.loss ≈ expected_loss
        @test state.gradients !== nothing
        @test model.weights != Float32[1 2; 3 4]

        calls[] = 0
        model2 = MockSNN(Float32[1 2; 3 4])
        _, positional_state = train_step!(model2, spikes, loss_fn, mock_step; rule=:ottt)
        @test calls[] == 1
        @test positional_state.loss ≈ expected_loss

        @test_throws ArgumentError train_step!(model, spikes, loss_fn; rule=:eprop)
        @test_throws ArgumentError train_step!(model, spikes, loss_fn; forward_fn=42, rule=:eprop)
    end

    @testset "train_step! ModelStep callable struct" begin
        model = MockSNN(Float32[1 2; 3 4])
        spikes = SpikeBatch(Float32[1 0 1; 0 1 1], nothing, nothing)
        loss_fn(output) = sum(output.logits)
        expected_loss = 20.0f0 / 3.0f0

        _, state = redirect_stdout(devnull) do
            train_step!(model, spikes, loss_fn; forward_fn=MockStep(), rule=:eprop)
        end
        @test state.loss ≈ expected_loss
        @test state.gradients !== nothing
    end

    @testset "e-prop and OTTT update weights and cut loss" begin
        Random.seed!(4)
        n_pre, n_out, T = 6, 3, 32
        W_true = Float32[0.8 0 0 0 0 0;
                         0 0.8 0 0 0 0;
                         0 0 0.8 0 0 0]
        spikes_mat = zeros(Float32, n_pre, T)
        spikes_mat[1, 1:2:T] .= 1
        spikes_mat[2, 2:3:T] .= 1
        spikes_mat[3, 1:4:T] .= 1
        rates = vec(mean(spikes_mat; dims=2))
        target = W_true * rates
        batch = SpikeBatch(spikes_mat, nothing, target)

        function rate_step(model, batch::SpikeBatch)
            r = vec(mean(batch.spikes; dims=2))
            return (logits = model.weights * r,)
        end
        loss_fn(output) = sum(abs2, output.logits .- target)
        opt = SynapticDistill.default_optimizer(0.05f0)

        function run_rule(rule)
            model = MockSNN(0.01f0 .* randn(Float32, n_out, n_pre))
            _, s0 = train_step!(model, batch, loss_fn; forward_fn=rate_step, rule=rule, optimizer=opt)
            loss0 = s0.loss
            W0 = copy(model.weights)
            traces = s0.traces
            local last = s0
            for _ in 1:40
                _, last = train_step!(model, batch, loss_fn;
                                      forward_fn=rate_step, rule=rule,
                                      optimizer=opt, traces=traces)
                traces = last.traces
            end
            return loss0, last.loss, W0, copy(model.weights), last
        end

        for rule in (:eprop, :ottt)
            loss0, loss1, W0, W1, last = run_rule(rule)
            @test last.gradients isa AbstractMatrix
            @test size(last.gradients) == (n_out, n_pre)
            @test last.traces isa TraceBatch
            @test W1 != W0
            @test loss1 < loss0
        end

        model = MockSNN(randn(Float32, n_out, n_pre))
        grads, tr = update_eprop!(model, batch, 1.0f0, (logits = zeros(Float32, n_out),);
                                  loss_fn = loss_fn)
        @test size(grads) == (n_out, n_pre)
        @test tr.traces.rule === :eprop
        grads2, _ = update_ottt!(model, batch, 1.0f0, (logits = zeros(Float32, n_out),);
                                 loss_fn = loss_fn, traces=tr)
        @test size(grads2) == (n_out, n_pre)
    end

    @testset "single-tick vector-of-vectors spikes" begin
        # `push!` into `[]` yields `Vector{Any}`; generic `reduce(hcat, ·)` would
        # return the inner vector instead of an `n_pre × 1` matrix.
        tick = Any[]
        push!(tick, Float32[1, 0, 1])
        S = SynapticDistill._spikes_as_matrix(tick)
        @test S isa AbstractMatrix
        @test size(S) == (3, 1)
        @test S == reshape(Float32[1, 0, 1], 3, 1)

        model = MockSNN(Float32[0.1 0.2 0.3; 0.4 0.5 0.6])
        batch = SpikeBatch(tick, nothing, nothing)
        output = (logits = zeros(Float32, 2),)
        grads, tr = update_eprop!(model, batch, 1.0f0, output)
        @test size(grads) == (2, 3)
        @test tr isa TraceBatch

        batch2 = SpikeBatch(Any[Float32[0, 1, 0]], nothing, nothing)
        grads2, _ = update_ottt!(model, batch2, 1.0f0, output; traces=tr)
        @test size(grads2) == (2, 3)
    end

    @testset "OTTT is per-timestep, not a rename of e-prop" begin
        Random.seed!(3)
        n_pre, n_out, T = 6, 3, 8
        S = Float32.(rand(0:1, n_pre, T))
        batch = SpikeBatch(S, nothing, nothing)
        model = MockSNN(0.1f0 .* randn(Float32, n_out, n_pre))

        # `n_out × T` logits + a loss that weights timesteps differently, so
        # ∂L/∂logits genuinely varies with t.
        w = Float32.(collect(1:T))
        out_mat = (logits = randn(Float32, n_out, T),)
        loss_mat = o -> sum(sum(abs2, o.logits; dims=1)[:] .* w)
        g_ottt, tr = update_ottt!(model, batch, 1.0f0, out_mat; loss_fn = loss_mat)
        @test tr.traces.time_resolved
        @test size(g_ottt) == (n_out, n_pre)

        out_vec = (logits = vec(sum(out_mat.logits; dims=2)),)
        loss_vec = o -> sum(abs2, o.logits)
        g_ep, _ = update_eprop!(model, batch, 1.0f0, out_vec; loss_fn = loss_vec)

        # The whole point: a time-resolved signal cannot be refactored into L ⊗ ȳ.
        @test !isapprox(g_ottt, g_ep; rtol = 1f-3)

        # A vector `logits` carries no per-timestep information, so OTTT must
        # collapse back onto e-prop exactly — and say so.
        g_deg, tr_deg = update_ottt!(model, batch, 1.0f0, out_vec; loss_fn = loss_vec)
        @test !tr_deg.traces.time_resolved
        @test isapprox(g_deg, g_ep; rtol = 1f-5)

        # Wrong column count is a clear error, not a silent broadcast.
        bad = (logits = randn(Float32, n_out, T + 1),)
        @test_throws DimensionMismatch update_ottt!(model, batch, 1.0f0, bad;
                                                    loss_fn = loss_mat)
    end

    @testset "ambiguous square spike layout is rejected" begin
        model = MockSNN(randn(Float32, 2, 4))
        square = SpikeBatch(Float32.(rand(0:1, 4, 4)), nothing, nothing)
        @test_throws ArgumentError SynapticDistill._spike_matrix(square, 4)
        @test_throws ArgumentError update_eprop!(model, square, 1.0f0,
                                                 (logits = zeros(Float32, 2),))

        # Non-square stays unambiguous in both orientations.
        @test size(SynapticDistill._spike_matrix(
            SpikeBatch(Float32.(rand(0:1, 4, 7)), nothing, nothing), 4)) == (4, 7)
        @test size(SynapticDistill._spike_matrix(
            SpikeBatch(Float32.(rand(0:1, 7, 4)), nothing, nothing), 4)) == (4, 7)

        # 1×1 is unambiguous: permutedims is a no-op, so both layouts coincide.
        one = SpikeBatch(reshape(Float32[1], 1, 1), nothing, nothing)
        @test SynapticDistill._spike_matrix(one, 1) == reshape(Float32[1], 1, 1)
        @test size(SynapticDistill._spike_matrix(
            SpikeBatch([[1.0f0]], nothing, nothing), 1)) == (1, 1)
        model1 = MockSNN(reshape(Float32[0.5], 1, 1))
        grads1, _ = update_eprop!(model1, one, 1.0f0, (logits = zeros(Float32, 1),))
        @test size(grads1) == (1, 1)
    end

end
