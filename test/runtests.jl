# SPDX-License-Identifier: MIT OR Apache-2.0

using Test
using SynapticDistill
using LinearAlgebra
using Random
using Statistics
using Zygote
using JSON3

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

    @testset "spikenaut_train sidecar (v3 live encoder + Dale + K-WTA)" begin
        script_src = read(joinpath(@__DIR__, "..", "scripts", "spikenaut_train.jl"), String)
        @test !occursin(r"(?m)^using SynapticDistill\b", script_src)
        @test occursin("mem_util_pct", script_src)
        @test occursin("sm_clock_mhz", script_src)
        @test occursin("74acdd0f", script_src)
        @test occursin("gpu-000000..138", script_src)
        @test occursin("parameters_output_weights.mem", script_src)
        # Encoder path must not *read* *_derived. The names may appear only
        # as forbidden-sensor refusals (exp-008).
        @test occursin("FORBIDDEN_SENSORS", script_src)
        @test occursin("Refusing *_derived", script_src)

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

        @testset "v3 state_telemetry encoder (exp-008..011)" begin
            # Fixture row 1: live fields + contradictory *_derived / forbidden
            # extras. Scaling must match frozen train constants, not derived.
            fixture = joinpath(@__DIR__, "fixtures", "state_telemetry_head.jsonl")
            rows = load_jsonl(fixture)
            @test length(rows) == 7
            rec = rows[1]
            @test is_state_telemetry(rec)
            @test is_forbidden_derived(rec)  # derived keys are present but unread

            stim = to_stimuli(rec)
            @test length(stim) == N_CHANNELS
            @test all(0 .<= stim .<= 1)
            # Frozen scales, sha lineage 74acdd0f:
            # mem_util 37.5 / 75 = 0.5
            # power at train min → 0
            # gpu_temp 34.5 / 69 = 0.5
            # sm (1545-180)/(2910-180) = 0.5
            # memclk at train min → 0
            @test stim[1] ≈ 0.5f0
            @test stim[2] ≈ 0f0
            @test stim[3] ≈ 0.5f0
            @test stim[4] ≈ 0.5f0
            @test stim[5] ≈ 0f0
            # Unused axons 5..15 (Julia 6:16) are unused width, not data.
            @test all(==(0f0), stim[6:16])

            # Changing only *_derived must not move any axon.
            poisoned = Dict(
                :mem_util_pct => 37.5,
                :power_w => 8.527000427246094,
                :gpu_temp_c => 34.5,
                :sm_clock_mhz => 1545,
                :mem_clock_mhz => 405,
                :hashrate_mh_derived => 0.0,
                :power_w_derived => 10.0,
                :gpu_temp_c_derived => 0.0,
                :reward_hint_derived => 0.0,
                :tick_rate => 0.0,
            )
            @test to_stimuli(poisoned) == stim

            # T=0 stays 0 — no impute from vram or a neighbour.
            idle = to_stimuli(rows[2])
            @test idle[3] == 0f0
            @test all(==(0f0), idle[6:16])

            # *_derived-only records (qubic_ticks_snn) are refused.
            derived_only = Dict(
                :tick_rate => 0.4333,
                :hashrate_mh_derived => 1.812488,
                :power_w_derived => 381.248828,
                :gpu_temp_c_derived => 72.187324,
                :reward_hint_derived => 0.812488,
            )
            @test !is_state_telemetry(derived_only)
            @test is_forbidden_derived(derived_only)
            @test_throws ErrorException to_stimuli(derived_only)
            @test_throws ErrorException assert_legal_sensors!([derived_only])

            qubic = load_jsonl(joinpath(@__DIR__, "fixtures", "qubic_ticks_snn_head.jsonl"))
            @test_throws ErrorException to_stimuli(qubic[1])
            @test_throws ErrorException assert_legal_sensors!(qubic)

            # Reward / readout use live power_w and gpu_temp_c, not *_derived.
            # Row 1: temp_u=0.5, power_u=0 → reward = 1 - 0.5 - 0 = 0.5
            @test sample_reward(rec) ≈ 0.5f0
            tgt = sample_readout_target(rec)
            @test tgt[2] ≈ 0.5f0          # live temp
            @test tgt[3] ≈ 0f0            # live power at train min
            # Derived would have been temp 75 / power 400 if those were read.
            hot = Dict(:gpu_temp_c => 69.0, :power_w => 302.8450012207031,
                       :mem_util_pct => 0, :sm_clock_mhz => 180, :mem_clock_mhz => 405,
                       :gpu_temp_c_derived => 0.0, :power_w_derived => 8.5)
            @test sample_reward(hot) < 0
            @test sample_readout_target(hot)[2] ≈ 1f0
            @test sample_readout_target(hot)[3] ≈ 1f0

            # Clamp after scale: values outside the train box stay in [0, 1].
            ood = Dict(:mem_util_pct => 90, :power_w => 0, :gpu_temp_c => 80,
                       :sm_clock_mhz => 100, :mem_clock_mhz => 20000)
            ood_stim = to_stimuli(ood)
            @test all(0 .<= ood_stim .<= 1)
            @test ood_stim[1] == 1f0
            @test ood_stim[3] == 1f0
            @test ood_stim[5] == 1f0

            # Live KEY presence, not value. JSON null on the first live
            # column must not refuse the row (exp-014). Null encodes as 0.
            null_first = JSON3.read(
                "{\"mem_util_pct\":null,\"power_w\":8.527000427246094," *
                "\"gpu_temp_c\":34.5,\"sm_clock_mhz\":1545,\"mem_clock_mhz\":405}"
            )
            @test is_state_telemetry(null_first)
            @test rec_has(null_first, :mem_util_pct)
            @test rec_get(null_first, :mem_util_pct) === nothing
            null_stim = to_stimuli(null_first)
            @test null_stim[1] == 0f0
            @test null_stim[2] ≈ 0f0
            @test null_stim[3] ≈ 0.5f0
            @test null_stim[4] ≈ 0.5f0
            @test null_stim[5] ≈ 0f0
            @test all(==(0f0), null_stim[6:16])
        end

        @testset "episode holdout (no row shuffle)" begin
            rows = load_jsonl(joinpath(@__DIR__, "fixtures", "state_telemetry_head.jsonl"))
            @test episode_index("gpu-000138") == 138
            @test episode_split("gpu-000000") === :train
            @test episode_split("gpu-000138") === :train
            @test episode_split("gpu-000139") === nothing
            @test episode_split("gpu-000150") === :val
            @test episode_split("gpu-000168") === :val
            @test episode_split("gpu-000169") === nothing
            @test episode_split("gpu-000170") === :test
            @test episode_split("gpu-000198") === :test

            tr = filter_split(rows, :train)
            va = filter_split(rows, :val)
            te = filter_split(rows, :test)
            @test length(tr) == 2 && all(r -> episode_index_of(r) == 0, tr)
            @test length(va) == 1 && episode_index_of(va[1]) == 150
            @test length(te) == 2
            @test episode_index_of(te[1]) == 180
            @test episode_index_of(te[2]) == 196
            # File order preserved; embargo 139 and 169 never appear.
            embargoed = [episode_index_of(r) for r in rows if episode_split(episode_index_of(r)) === nothing]
            @test embargoed == [139, 169]
            @test [episode_index_of(r) for r in tr] == [0, 0]
            # Missing episode_id is dropped, not invented.
            @test filter_split([Dict(:mem_util_pct => 1)], :train) == []

            # A row from another split still breaks adjacency: train ep0 /
            # test ep170 / train ep0 would put the two ep0 fragments next to
            # each other in the output, so the `ep !== prev_ep` reset would
            # never fire between them. Refuse it.
            interleaved = [
                Dict(:episode_id => "gpu-000000", :mem_util_pct => 1),
                Dict(:episode_id => "gpu-000170", :mem_util_pct => 2),
                Dict(:episode_id => "gpu-000000", :mem_util_pct => 3),
            ]
            @test_throws ErrorException filter_split(interleaved, :train)
            # Six digits or nothing: `gpu-150` is unparseable, not episode 150.
            @test episode_index("gpu-150") === nothing

            # Health must use test, never the already-filtered train split.
            health_rows = require_test_split(rows)
            @test length(health_rows) == 2
            @test all(r -> episode_split(episode_index_of(r)) === :test, health_rows)
            @test episode_index_of(health_rows[1]) == 180
            @test episode_index_of(health_rows[2]) == 196
            @test health_rows != tr
            @test_throws ErrorException require_test_split(tr)
            @test_throws ErrorException require_test_split([Dict(:mem_util_pct => 1)])
            @test require_train_cli_split(:train) === :train
            @test_throws ErrorException require_train_cli_split(:val)
            @test_throws ErrorException require_train_cli_split(:test)
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
                @test occursin("outgoing", model)
                @test occursin("v3_state_telemetry", model)
                @test occursin("mem_util_pct", model)
                @test occursin("74acdd0f", model)
                @test occursin("5:15", model)
                @test length(paths) == 5
            end

            # Health eval is k=none: a strong drive can fire more than K_WTA.
            Random.seed!(11)
            eval_bank = LIFBank()
            eval_bank.weights .= 0.4f0
            n_none = tick!(eval_bank, fill(1f0, N_CHANNELS), 0f0, nothing; k=nothing, learn=false)
            @test n_none == N_NEURONS
            eval_bank.v .= 0.42f0
            v_before = copy(eval_bank.v)
            spikes_before = copy(eval_bank.spikes)
            h = health_eval(eval_bank, [Dict(
                :episode_id => "gpu-000170",
                :mem_util_pct => 75, :power_w => 302.8450012207031,
                :gpu_temp_c => 69, :sm_clock_mhz => 2910, :mem_clock_mhz => 14801,
            )])
            @test h.k == "none"
            @test h.n == 1
            @test haskey(h, :cofire)
            @test h.cofire >= 0
            # health_eval must not overwrite the post-train membrane that
            # export_artifacts writes into snn_model.json (exp-014).
            @test eval_bank.v == v_before
            @test eval_bank.spikes == spikes_before
            mktempdir() do dir
                export_artifacts(eval_bank, dir)
                model = JSON3.read(read(joinpath(dir, "snn_model.json"), String))
                @test all(n -> Float32(n.membrane_potential) == 0.42f0, model.neurons)
            end

            # Mean pairwise cofire (cosine, silent neurons excluded).
            # All-16 every tick → 1.0. Four disjoint k=4 groups → 0.20.
            n = N_NEURONS
            counts16 = fill(4, n)
            both16 = zeros(Int, n, n)
            for i in 1:(n - 1), j in (i + 1):n
                both16[i, j] = 4
            end
            @test mean_pairwise_cofire(counts16, both16) ≈ 1.0
            counts_k4 = ones(Int, n)
            both_k4 = zeros(Int, n, n)
            for g in 0:3
                members = (4g + 1):(4g + 4)
                for i in members, j in members
                    i < j && (both_k4[i, j] = 1)
                end
            end
            @test mean_pairwise_cofire(counts_k4, both_k4) ≈ 0.2
            # 11 lockstep + 5 silent → 1.0, not C(11,2)/C(16,2).
            counts11 = [fill(3, 11); zeros(Int, 5)]
            both11 = zeros(Int, n, n)
            for i in 1:10, j in (i + 1):11
                both11[i, j] = 3
            end
            @test mean_pairwise_cofire(counts11, both11) ≈ 1.0
            @test mean_pairwise_cofire(zeros(Int, n), zeros(Int, n, n)) == 0.0
        end

        @testset "parquet ingest is refused (no silent converter)" begin
            mktempdir() do dir
                pq = joinpath(dir, "train-00000.parquet")
                write(pq, "not a real parquet")
                @test_throws ErrorException load_data(pq)
                @test_throws ErrorException load_data(dir)
            end
        end

        @testset "health_eval is test split, not train (exp-013)" begin
            fixture = joinpath(@__DIR__, "fixtures", "state_telemetry_head.jsonl")
            rows = load_jsonl(fixture)
            mktempdir() do dir
                # Train-only JSONL must error — not silently print train health.
                only_train = joinpath(dir, "train_only.jsonl")
                write(only_train, join(readlines(fixture)[1:2], "\n") * "\n")
                @test_throws ErrorException main([only_train, "1", joinpath(dir, "out-train")])
                @test_throws ErrorException main([fixture, "1", joinpath(dir, "out-val"), "val"])
                @test_throws ErrorException main([fixture, "1", joinpath(dir, "out-test"), "test"])

                # Full fixture has test episodes: train on train, eval n == test n.
                out = joinpath(dir, "out-full")
                bank = main([fixture, "1", out, "train"])
                @test bank isa LIFBank
                te = require_test_split(rows)
                h = health_eval(bank, te)
                @test h.n == length(te) == 2
                @test h.n != 7
                @test h.k == "none"
                @test haskey(h, :cofire)
            end
        end
    end

end
