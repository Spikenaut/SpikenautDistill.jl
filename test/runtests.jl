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
    @testset "spikenaut_train sidecar (signed E/I + K-WTA + Q8.8)" begin
        script_src = read(joinpath(@__DIR__, "..", "scripts", "spikenaut_train.jl"), String)
        @test !occursin(r"(?m)^using SynapticDistill\b", script_src)
        @test occursin("reward_hint_derived", script_src)
        @test occursin("parameters_output_weights.mem", script_src)

        # Include the standalone sidecar without running main().
        include(joinpath(@__DIR__, "..", "scripts", "spikenaut_train.jl"))

        @testset "q88_signed two's complement" begin
            @test q88_signed(0) == "0000"
            @test q88_signed(1) == "0100"
            @test q88_signed(0.75) == "00C0"
            @test q88_signed(-7 / 256) == "FFF9"
            @test q88_signed(-1) == "FF00"
            @test q88_signed(DECAY) == q88_signed(0.85f0)
        end

        @testset "27k telemetry encoder" begin
            rec = Dict(
                :timestamp => "2026-03-20T08:55:24+00:00",
                :tick => 46538099,
                :tick_rate => 0.4333,
                :qubic_tick_trace => 0.0,
                :hashrate_mh_derived => 1.812488,
                :power_w_derived => 381.248828,
                :gpu_temp_c_derived => 72.187324,
                :reward_hint_derived => 0.812488,
            )
            stim = to_stimuli(rec)
            @test length(stim) == N_CHANNELS
            @test all(0 .<= stim .<= 1)
            @test is_telemetry(rec)

            r = sample_reward(rec)
            @test -1 ≤ r ≤ 1
            # High hint (0.81) minus mild thermal/power pain — not clamped to [0, 1] only.
            pain = sample_reward(Dict(
                :reward_hint_derived => 0.0,
                :gpu_temp_c_derived => 75.0,
                :power_w_derived => 400.0,
            ))
            @test pain < 0

            enc = StimEncoder()
            s1 = to_stimuli(rec, enc)
            rec2 = merge(rec, Dict(:hashrate_mh_derived => 1.0, :reward_hint_derived => 0.0))
            s2 = to_stimuli(rec2, enc)
            @test s1 != s2

            # Delta channels must actually track history. `s1 != s2` alone
            # passes on the level channels even when `prev` aliases `enc.prev`
            # and every delta reads `cur - cur`.
            @test all(≈(0.5f0), s1[7:11])   # first tick: no history, neutral
            @test s1[14] == 0f0
            @test s2[9] < 0.5f0             # hashrate 1.812 -> 1.0, so down
            @test s2[14] > 0f0              # hash_drop registers the fall

            fixture = joinpath(@__DIR__, "fixtures", "qubic_ticks_snn_head.jsonl")
            rows = load_jsonl(fixture)
            @test length(rows) == 4
            @test length(to_stimuli(rows[1])) == 16
            @test sample_reward(rows[4]) < 0
        end

        @testset "legacy spikes still work" begin
            stim = to_stimuli(Dict(:spikes => [0.2, 0.8, 1.5]))
            @test stim[1] == 0.2f0
            @test stim[2] == 0.8f0
            @test stim[3] == 1.0f0
            @test all(stim[4:end] .== 0)
        end

        @testset "K-WTA + Dale + mixed-sign export" begin
            Random.seed!(29)
            bank = LIFBank()
            # Dale lives on OUTGOING weights: readout column i is neuron i's
            # projection. Incoming weights carry no sign constraint, which is
            # what lets inhibitory neurons be driven to threshold at all.
            @test all(>=(0), bank.readout[:, 1:N_EXC])
            @test all(<=(0), bank.readout[:, INHIB_ROWS])

            stim = fill(0.95f0, N_CHANNELS)
            nspk = tick!(bank, stim, 0.4f0, Float32[0.8, 0.6, 0.7])
            @test nspk <= K_WTA
            @test count(bank.spikes) <= K_WTA

            # Drive LTD + signed reward; Dale must hold on the readout throughout.
            inhib_spikes = 0
            for _ in 1:400
                tick!(bank, rand(Float32, N_CHANNELS), randn(Float32), rand(Float32, 3))
                inhib_spikes += count(bank.spikes[INHIB_ROWS])
            end
            @test all(>=(0), bank.readout[:, 1:N_EXC])
            @test all(<=(0), bank.readout[:, INHIB_ROWS])

            # The regression this change exists for: with Dale on incoming
            # weights, rows 13:16 sat near -30 against a +1 threshold and fired
            # exactly 0 times, so the exported "E/I" model had no inhibition.
            @test inhib_spikes > 0

            @test minimum(bank.weights) < 0 < maximum(bank.weights)
            @test std(bank.weights) > 0.01f0

            mktempdir() do dir
                paths = export_artifacts(bank, dir)
                @test isfile(joinpath(dir, "parameters_output_weights.mem"))
                @test countlines(joinpath(dir, "parameters_output_weights.mem")) == 48
                @test countlines(joinpath(dir, "parameters_weights.mem")) == 256
                @test countlines(joinpath(dir, "parameters.mem")) == 16
                @test countlines(joinpath(dir, "parameters_decay.mem")) == 16
                # Keep-factor decay must stay 0.85 → 00D9 or 00DA (0.85*256=217.6).
                decay_hex = strip(read(joinpath(dir, "parameters_decay.mem"), String))
                @test occursin("00D9", decay_hex) || occursin("00DA", decay_hex)
                out_hex = read(joinpath(dir, "parameters_output_weights.mem"), String)
                @test occursin(r"^[0-9A-F]{4}$"m, out_hex)

                # Emission ORDER, not just line count. Both memories are
                # neuron-major with the second index varying fastest. Line
                # counts alone cannot catch a transposed write, and a
                # column-major generator silently produces exactly that.
                wlines = readlines(joinpath(dir, "parameters_weights.mem"))
                @test wlines[1] == q88_signed(bank.weights[1, 1])
                @test wlines[2] == q88_signed(bank.weights[1, 2])
                @test wlines[N_CHANNELS + 1] == q88_signed(bank.weights[2, 1])
                @test wlines[end] == q88_signed(bank.weights[N_NEURONS, N_CHANNELS])

                olines = readlines(joinpath(dir, "parameters_output_weights.mem"))
                @test olines[1] == q88_signed(bank.readout[1, 1])
                @test olines[2] == q88_signed(bank.readout[2, 1])
                @test olines[N_OUTPUTS + 1] == q88_signed(bank.readout[1, 2])
                @test olines[end] == q88_signed(bank.readout[N_OUTPUTS, N_NEURONS])
                # Signed encoder must be able to emit FFF9 (regression vs unsigned clamp).
                @test q88_signed(-7 / 256) == "FFF9"
                model = read(joinpath(dir, "snn_model.json"), String)
                @test occursin("keep", model)
                @test occursin("80:20", model)
                @test length(paths) == 5
            end
        end
    end

end
