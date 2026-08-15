#!/usr/bin/env julia
# SPDX-License-Identifier: MIT OR Apache-2.0
# spikenaut_train.jl — Spikenaut LIF Trainer (standalone sidecar)
#
# Loads temporal event-stream JSONL *or* qubic_ticks_snn telemetry JSONL,
# runs LIF + Dale 80:20 + K-WTA + reward-modulated STDP / e-prop,
# then writes snn_model.json + signed Q8.8 .mem files for
# rmems/Spikenaut-SNN `dataset/merged_v2/`.
#
# Standalone: JSON3 + stdlib only. Does **not** `using SynapticDistill`
# (library `update_eprop!` / `update_ottt!` are still stubs).
#
# Usage:
#   julia scripts/spikenaut_train.jl <data_path> [epochs] [out_dir]
#   julia scripts/spikenaut_train.jl \
#     /home/raulmc/Spikenaut-Vault/Spikenaut-SNN-Telemetry/full_data/qubic_ticks_snn.jsonl \
#     20 /tmp/spikenaut-out
#
# Prefer the ~27k `qubic_ticks_snn.jsonl`. The 8-record `fresh_sync` sample
# produces degenerate / monotonic hidden weights.

using JSON3, LinearAlgebra, Printf, Random, Statistics

# ── Config ────────────────────────────────────────────────────────────
const N_NEURONS    = 16
const N_CHANNELS   = 16
const N_OUTPUTS    = 3
const N_INHIB      = 4
const N_EXC        = N_NEURONS - N_INHIB          # 12 — Dale 80:20
const INHIB_ROWS   = (N_EXC + 1):N_NEURONS        # 13:16
const K_WTA        = 4
const DECAY        = 0.85f0                       # keep factor, not leak
const STDP_LTP     = 0.01f0
const STDP_LTD     = 0.005f0
const EPROP_LR     = 0.002f0
const READOUT_LR   = 0.01f0
const W_MIN        = -1.0f0
const W_MAX        = 2.0f0
const ROW_L2_CAP   = 2.0f0
const THRESH_INIT  = 1.0f0
const TRACE_LAMBDA = 0.85f0
const MIN_TRAIN_N  = 100
const DEFAULT_27K  = "/home/raulmc/Spikenaut-Vault/Spikenaut-SNN-Telemetry/full_data/qubic_ticks_snn.jsonl"

# Physical scales observed on qubic_ticks_snn (27,430 rows).
# Stimuli are Poisson rates in [0, 1]; deltas are folded through 0.5 + 0.5 tanh.
const TICK_RATE_SCALE   = 0.5333f0
const HASHRATE_SCALE    = 2.0f0
const POWER_MIN         = 300.0f0
const POWER_SPAN        = 100.0f0
const TEMP_MIN          = 60.0f0
const TEMP_SPAN         = 15.0f0
const THERMAL_COMFORT_C = 70.0f0
const POWER_BUDGET_W    = 380.0f0

# ── LIF State ─────────────────────────────────────────────────────────
mutable struct LIFBank
    v        ::Vector{Float32}   # membrane potentials
    thresh   ::Vector{Float32}   # thresholds
    weights  ::Matrix{Float32}   # [N_NEURONS × N_CHANNELS]
    decay    ::Vector{Float32}   # per-neuron keep factor
    spikes   ::Vector{Bool}
    pre_tr   ::Vector{Float32}   # OTTT presynaptic traces
    elig     ::Matrix{Float32}   # eligibility traces [N × CH]
    readout  ::Matrix{Float32}   # [N_OUTPUTS × N_NEURONS]
end

function init_hidden_weights()
    W = randn(Float32, N_NEURONS, N_CHANNELS) .* 0.08f0
    @inbounds for i in 1:N_EXC
        for j in 1:N_CHANNELS
            W[i, j] = abs(W[i, j]) + 0.04f0
        end
    end
    @inbounds for i in INHIB_ROWS
        for j in 1:N_CHANNELS
            W[i, j] = -abs(W[i, j]) - 0.04f0
        end
    end
    return W
end

function LIFBank()
    LIFBank(
        zeros(Float32, N_NEURONS),
        fill(THRESH_INIT, N_NEURONS),
        init_hidden_weights(),
        fill(DECAY, N_NEURONS),
        falses(N_NEURONS),
        zeros(Float32, N_CHANNELS),
        zeros(Float32, N_NEURONS, N_CHANNELS),
        randn(Float32, N_OUTPUTS, N_NEURONS) .* 0.05f0,
    )
end

# ── Record field access (JSON3.Object uses Symbol keys) ───────────────
function rec_get(sample, names::Symbol...)
    for name in names
        if haskey(sample, name)
            return sample[name]
        end
        s = String(name)
        if haskey(sample, s)
            return sample[s]
        end
    end
    return nothing
end

function rec_f32(sample, default::Float32, names::Symbol...)
    v = rec_get(sample, names...)
    v === nothing && return default
    return Float32(v)
end

# ── Stimulus / reward from 27k telemetry or legacy spike rows ─────────
mutable struct StimEncoder
    prev::Vector{Float32}   # [tick_rate, trace, hashrate, power, temp, hint]
    initialized::Bool
end
StimEncoder() = StimEncoder(zeros(Float32, 6), false)

function _unit01(x::Float32)
    return clamp(x, 0f0, 1f0)
end

function _delta01(cur::Float32, prev::Float32, scale::Float32)
    # Map signed delta → Poisson rate in (0, 1).
    return Float32(0.5 + 0.5 * tanh((cur - prev) / max(scale, 1f-6)))
end

"""
    is_telemetry(sample) -> Bool

True for `qubic_ticks_snn` rows (`reward_hint_derived`, `hashrate_mh_derived`,
`gpu_temp_c_derived`, `tick_rate`, …). Those rows have **no** `spikes`/`inputs`.
"""
function is_telemetry(sample)
    rec_get(sample, :reward_hint_derived, :hashrate_mh_derived,
            :gpu_temp_c_derived, :power_w_derived, :tick_rate) !== nothing
end

function raw_telemetry(sample)
    tick_rate = rec_f32(sample, 0f0, :tick_rate)
    trace     = rec_f32(sample, 0f0, :qubic_tick_trace)
    hashrate  = rec_f32(sample, 0f0, :hashrate_mh_derived)
    power     = rec_f32(sample, POWER_MIN, :power_w_derived)
    temp      = rec_f32(sample, TEMP_MIN, :gpu_temp_c_derived)
    hint      = rec_f32(sample, 0.5f0, :reward_hint_derived, :reward_hint, :reward)
    return Float32[tick_rate, trace, hashrate, power, temp, hint]
end

"""
    to_stimuli(sample, enc=nothing) -> Vector{Float32}

16-channel Poisson rates.

- Legacy: `spikes` / `inputs` (clamped to [0, 1], padded/truncated to 16).
- Telemetry (`qubic_ticks_snn`): 6 rate-coded physical channels, 5 first
  differences, 5 pain / efficiency / composite channels.

Does **not** require `spikes`/`inputs` — that path crashed on the 27k file.
"""
function to_stimuli(sample, enc::Union{StimEncoder,Nothing}=nothing)
    raw = rec_get(sample, :spikes, :inputs)
    if raw !== nothing
        stim = zeros(Float32, N_CHANNELS)
        n = min(length(raw), N_CHANNELS)
        for i in 1:n
            stim[i] = clamp(Float32(raw[i]), 0f0, 1f0)
        end
        return stim
    end

    is_telemetry(sample) || error(
        "Sample is missing `spikes`/`inputs` and is not qubic_ticks_snn telemetry " *
        "(expected reward_hint_derived / hashrate_mh_derived / gpu_temp_c_derived / tick_rate)"
    )

    cur = raw_telemetry(sample)
    prev = if enc !== nothing && enc.initialized
        enc.prev
    else
        cur
    end
    if enc !== nothing
        enc.prev .= cur
        enc.initialized = true
    end

    tick_rate, trace, hashrate, power, temp, hint = cur
    p_tick, p_trace, p_hash, p_power, p_temp, p_hint = prev

    thermal_pain = _unit01((temp - THERMAL_COMFORT_C) / 10f0)
    power_pain   = _unit01((power - POWER_BUDGET_W) / 50f0)
    hash_drop    = _unit01((p_hash - hashrate) / HASHRATE_SCALE)
    efficiency   = _unit01(hashrate / max(power / 200f0, 1f-3))
    composite    = _unit01(hint * (1f0 - 0.5f0 * thermal_pain - 0.5f0 * power_pain))

    return Float32[
        _unit01(tick_rate / TICK_RATE_SCALE),          # 1
        _unit01(trace),                                # 2
        _unit01(hashrate / HASHRATE_SCALE),            # 3
        _unit01((power - POWER_MIN) / POWER_SPAN),     # 4
        _unit01((temp - TEMP_MIN) / TEMP_SPAN),        # 5
        _unit01(hint),                                 # 6
        _delta01(tick_rate, p_tick, 0.1f0),            # 7
        _delta01(trace, p_trace, 0.05f0),              # 8
        _delta01(hashrate, p_hash, 0.2f0),             # 9
        _delta01(power, p_power, 10f0),                # 10
        _delta01(temp, p_temp, 2f0),                   # 11
        thermal_pain,                                  # 12
        power_pain,                                    # 13
        hash_drop,                                     # 14
        efficiency,                                    # 15
        composite,                                     # 16
    ]
end

"""
    sample_reward(sample) -> Float32

Signed learning signal in `[-1, 1]` (not `[0, 1]`).

Looks up `reward_hint_derived` first (27k schema), then legacy
`reward` / `target_reward` / `reward_hint`. Subtracts thermal and
power pain so the trainer can depress synapses.
"""
function sample_reward(sample)::Float32
    hint = rec_f32(sample, 0.5f0,
                   :reward_hint_derived, :reward, :target_reward, :reward_hint)
    temp  = rec_get(sample, :gpu_temp_c_derived)
    power = rec_get(sample, :power_w_derived)
    thermal_pain = temp === nothing ? 0f0 :
        _unit01((Float32(temp) - THERMAL_COMFORT_C) / 10f0)
    power_pain = power === nothing ? 0f0 :
        _unit01((Float32(power) - POWER_BUDGET_W) / 50f0)
    r = 2f0 * (hint - 0.5f0) - thermal_pain - 0.5f0 * power_pain
    return clamp(r, -1f0, 1f0)
end

function sample_readout_target(sample)::Vector{Float32}
    hint  = rec_f32(sample, 0.5f0,
                    :reward_hint_derived, :reward, :target_reward, :reward_hint)
    temp  = rec_f32(sample, THERMAL_COMFORT_C, :gpu_temp_c_derived)
    power = rec_f32(sample, POWER_BUDGET_W, :power_w_derived)
    return Float32[
        _unit01(hint),
        _unit01((temp - TEMP_MIN) / TEMP_SPAN),
        _unit01((power - POWER_MIN) / POWER_SPAN),
    ]
end

# ── Fast-sigmoid surrogate gradient ───────────────────────────────────
@inline surrogate(v, θ) = 1f0 / (1f0 + abs(10f0 * (v - θ)))^2

"""
    apply_kwta!(spikes, v, k)

Keep the `k` highest-`v` firers; silence the rest. Membrane of losers
is left intact so they can compete on the next tick.
"""
function apply_kwta!(spikes::AbstractVector{Bool}, v::AbstractVector, k::Int)
    nfire = count(spikes)
    nfire <= k && return spikes
    idx = findall(spikes)
    order = sortperm(view(v, idx); rev=true)
    fill!(spikes, false)
    @inbounds for t in 1:k
        spikes[idx[order[t]]] = true
    end
    return spikes
end

function apply_dale!(W::AbstractMatrix)
    @inbounds for i in 1:N_EXC
        for j in 1:size(W, 2)
            W[i, j] < 0 && (W[i, j] = 0f0)
        end
    end
    @inbounds for i in INHIB_ROWS
        for j in 1:size(W, 2)
            W[i, j] > 0 && (W[i, j] = 0f0)
        end
    end
    return W
end

function scale_rows_l2!(W::AbstractMatrix, cap::Float32)
    @inbounds for i in 1:size(W, 1)
        nrm = norm(view(W, i, :))
        if nrm > cap
            view(W, i, :) .*= cap / nrm
        end
    end
    return W
end

# ── One training tick ─────────────────────────────────────────────────
function tick!(bank::LIFBank, stim::Vector{Float32}, reward::Float32,
               target::Union{Vector{Float32},Nothing}=nothing)
    # 1. Poisson pre-spikes
    pre = Float32.(rand(Float32, N_CHANNELS) .< stim)

    # 2. OTTT presynaptic trace
    bank.pre_tr .= TRACE_LAMBDA .* bank.pre_tr .+ pre
    pre_snap = copy(bank.pre_tr)

    # 3. LIF forward: decay is a **keep** factor (Rust leak = 1 - keep).
    input = bank.weights * stim
    bank.v .= bank.decay .* bank.v .+ input

    # 4. Fire, then K-WTA, then reset winners only
    bank.spikes .= bank.v .>= bank.thresh
    apply_kwta!(bank.spikes, bank.v, K_WTA)
    bank.v[bank.spikes] .= 0f0

    # 5. STDP: LTP on co-activation, LTD when pre fired and post did not
    @inbounds for i in 1:N_NEURONS
        if bank.spikes[i]
            bank.weights[i, :] .+= STDP_LTP .* pre_snap
        else
            bank.weights[i, :] .-= STDP_LTD .* pre
        end
    end

    # 6. E-prop eligibility + signed reward (reward may be negative)
    @inbounds for i in 1:N_NEURONS
        dz = bank.spikes[i] ? 1f0 : surrogate(bank.v[i], bank.thresh[i])
        bank.elig[i, :] .= TRACE_LAMBDA .* bank.elig[i, :] .+ pre_snap .* dz
        bank.weights[i, :] .+= reward .* bank.elig[i, :] .* EPROP_LR
    end

    # 7. 16×3 readout: predict (hint, temp, power) from this tick's spikes
    if target !== nothing
        s = Float32.(bank.spikes)
        pred = bank.readout * s
        err = target .- pred
        bank.readout .+= (READOUT_LR * reward) .* (err * s')
    end

    # 8. Dale + signed L2 cap (do **not** divide rows by sum — that
    #    destroyed sign and collapsed every row onto the same simplex).
    apply_dale!(bank.weights)
    scale_rows_l2!(bank.weights, ROW_L2_CAP)
    clamp!(bank.weights, W_MIN, W_MAX)
    apply_dale!(bank.weights)

    sum(bank.spikes)
end

# ── Load JSONL ────────────────────────────────────────────────────────
function load_jsonl(path)
    samples = []
    open(path) do f
        for line in eachline(f)
            isempty(strip(line)) && continue
            try
                rec = JSON3.read(line)
                sample = if haskey(rec, :sensor_stream)
                    rec[:sensor_stream]
                elseif haskey(rec, "sensor_stream")
                    rec["sensor_stream"]
                elseif haskey(rec, :sample)
                    rec[:sample]
                elseif haskey(rec, "sample")
                    rec["sample"]
                else
                    rec
                end
                push!(samples, sample)
            catch
            end
        end
    end
    samples
end

# ── Load chunked directory ────────────────────────────────────────────
function load_chunked_dir(dir_path)
    samples = []
    chunk_files = String[]
    for entry in readdir(dir_path)
        full_path = joinpath(dir_path, entry)
        isfile(full_path) || continue
        if occursin(r"chunk", entry) && !endswith(entry, ".md") && !endswith(entry, ".json")
            push!(chunk_files, full_path)
        end
    end

    sort!(chunk_files)

    if isempty(chunk_files)
        @warn "No chunk files found in directory: $dir_path"
        return samples
    end

    println("Loading $(length(chunk_files)) chunk files...")
    for (i, chunk_file) in enumerate(chunk_files)
        print("  [$i/$(length(chunk_files))] Loading $(basename(chunk_file))... ")
        chunk_samples = load_jsonl(chunk_file)
        append!(samples, chunk_samples)
        println("$(length(chunk_samples)) records")
    end

    samples
end

function load_data(data_path)
    if isdir(data_path)
        load_chunked_dir(data_path)
    elseif isfile(data_path)
        load_jsonl(data_path)
    else
        error("Data path not found: $data_path")
    end
end

# ── Signed two's-complement Q8.8 ──────────────────────────────────────
"""
    q88_signed(v) -> String

Four uppercase hex digits, two's complement Q8.8.
`mod` (not `rem`/`%`) is required so negatives become `FFF9`, not `-7`.
"""
function q88_signed(v)
    q = clamp(round(Int, Float64(v) * 256), -32768, 32767)
    uppercase(string(UInt16(mod(q, 65536)), base=16, pad=4))
end

function write_mem(path, values)
    open(path, "w") do f
        for v in values
            println(f, q88_signed(v))
        end
    end
end

function export_artifacts(bank::LIFBank, out_dir::AbstractString)
    mkpath(out_dir)
    neurons_json = [
        Dict(
            "threshold"          => bank.thresh[i],
            "decay_rate"         => bank.decay[i],
            "membrane_potential" => bank.v[i],
            "weights"            => collect(bank.weights[i, :]),
            "output_weights"     => collect(bank.readout[:, i]),
            "inhibitory"         => i in INHIB_ROWS,
            "last_spike"         => false,
        )
        for i in 1:N_NEURONS
    ]
    open(joinpath(out_dir, "snn_model.json"), "w") do f
        JSON3.write(f, Dict(
            "neurons"        => neurons_json,
            "source"         => "spikenaut_julia",
            "ei_ratio"       => "80:20",
            "k_wta"          => K_WTA,
            "q88"            => "signed",
            "decay_semantics"=> "keep",
            "n_outputs"      => N_OUTPUTS,
        ))
    end

    write_mem(joinpath(out_dir, "parameters.mem"), bank.thresh)
    open(joinpath(out_dir, "parameters_weights.mem"), "w") do f
        for i in 1:N_NEURONS, ch in 1:N_CHANNELS
            println(f, q88_signed(bank.weights[i, ch]))
        end
    end
    write_mem(joinpath(out_dir, "parameters_decay.mem"), bank.decay)
    # 48 signed values, neuron-major: 16 neurons × 3 readout heads
    open(joinpath(out_dir, "parameters_output_weights.mem"), "w") do f
        for i in 1:N_NEURONS, o in 1:N_OUTPUTS
            println(f, q88_signed(bank.readout[o, i]))
        end
    end

    return (
        joinpath(out_dir, "snn_model.json"),
        joinpath(out_dir, "parameters.mem"),
        joinpath(out_dir, "parameters_weights.mem"),
        joinpath(out_dir, "parameters_decay.mem"),
        joinpath(out_dir, "parameters_output_weights.mem"),
    )
end

# ── Main ──────────────────────────────────────────────────────────────
function main(args=ARGS)
    length(args) >= 1 || error(
        "Usage: julia scripts/spikenaut_train.jl <data_path> [epochs] [out_dir]\n" *
        "  data_path: prefer qubic_ticks_snn.jsonl (~27430 rows), not fresh_sync (8).\n" *
        "  example:   julia scripts/spikenaut_train.jl $DEFAULT_27K 20 /tmp/spikenaut-out"
    )
    data_path = args[1]
    epochs    = length(args) >= 2 ? parse(Int, args[2]) : 20
    out_dir   = length(args) >= 3 ? args[3] : "out_train"

    isdir(data_path) || isfile(data_path) || error("Data path not found: $data_path")
    mkpath(out_dir)

    println("=== Spikenaut Julia Trainer ===")
    println("Data   : $data_path")
    println("Epochs : $epochs")
    println("Out    : $out_dir")
    println("Dale   : $N_EXC excitatory / $N_INHIB inhibitory; K-WTA k=$K_WTA")
    println("Decay  : keep=$DECAY  (Rust leak = $(1 - DECAY))")

    print("Loading samples... ")
    samples = load_data(data_path)
    println("$(length(samples)) total records")
    isempty(samples) && error("No valid samples found.")
    if length(samples) < MIN_TRAIN_N
        @warn "Only $(length(samples)) records — monotonic / collapsed weights are likely. Prefer $DEFAULT_27K (~27430)."
    end

    bank = LIFBank()
    enc  = StimEncoder()

    for epoch in 1:epochs
        total_reward = 0f0
        total_spikes = 0
        max_spikes   = 0

        fill!(bank.v, 0f0)
        enc.initialized = false

        t0 = time()
        for sample in samples
            stim   = to_stimuli(sample, enc)
            reward = sample_reward(sample)
            target = sample_readout_target(sample)
            nspk   = tick!(bank, stim, reward, target)
            total_spikes += nspk
            max_spikes    = max(max_spikes, nspk)
            total_reward += reward
        end
        elapsed = time() - t0

        n = length(samples)
        avg_r  = total_reward / n
        s_rate = total_spikes / (n * N_NEURONS)
        w_mean = mean(bank.weights)
        w_std  = std(bank.weights)
        w_min  = minimum(bank.weights)
        w_max  = maximum(bank.weights)
        n_inh  = count(i -> all(<=(0), view(bank.weights, i, :)), INHIB_ROWS)
        ms_tick = elapsed * 1000 / n

        @printf("Epoch %3d/%d | reward=%+.4f | spike_rate=%.3f | max_spk=%d | w=%+.4f±%.4f [%+.3f,%+.3f] | inhib_rows=%d | %.3fms/tick\n",
                epoch, epochs, avg_r, s_rate, max_spikes, w_mean, w_std, w_min, w_max, n_inh, ms_tick)
    end

    paths = export_artifacts(bank, out_dir)
    println("\nExported:")
    for p in paths
        println("  $p")
    end
    println("Hidden weights: min=$(minimum(bank.weights)) max=$(maximum(bank.weights)) std=$(std(bank.weights))")
    println("SUCCESS: Spikenaut trained (signed E/I + K-WTA + signed Q8.8).")
    return bank
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
