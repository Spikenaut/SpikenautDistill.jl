#!/usr/bin/env julia
# SPDX-License-Identifier: MIT OR Apache-2.0
# spikenaut_train.jl — Spikenaut LIF Trainer (standalone sidecar)
#
# Loads temporal event-stream JSONL *or* v3 `state_telemetry` JSONL
# (five live sensors). Runs LIF + outgoing Dale 80:20 + K-WTA +
# reward-modulated STDP / e-prop, then writes snn_model.json + signed
# Q8.8 .mem files.
#
# Standalone: JSON3 + stdlib only. Does **not** `using SynapticDistill`
# (library `update_eprop!` / `update_ottt!` are still stubs).
#
# Legal live columns (Spikenaut Scientist exp-008; survive train/val/test,
# not constant, not all-null, not identity). Axons 0..4 (Julia 1..5):
#   mem_util_pct, power_w, gpu_temp_c, sm_clock_mhz, mem_clock_mhz
# Axons 5..15 (Julia 6..16) are unused width, held at 0 — not fake channels.
#
# Forbidden: *_derived (closed form of tick_rate; 0 mismatches / 27430),
# fan_speed_pct, vddcr_gfx_v (constant on val), vram_temp_c (gpu_temp+8
# except idle 0), step_idx, first-difference inventions, fabricated ts_utc.
#
# Frozen minmax (v3 state_telemetry train, sha lineage 74acdd0f).
# Clamp to [0, 1] after scale. Do not refit on val/test.
#
# Episode holdout (never shuffle rows across episodes; embargo 139, 169):
#   train gpu-000000..138   val gpu-000140..168   test gpu-000170..198
#
# Usage:
#   julia scripts/spikenaut_train.jl <data_path> [epochs] [out_dir] [split]
#   split must be train (default). val/test error — no learn=true on holdout.
#   julia scripts/spikenaut_train.jl \
#     /path/to/state_telemetry.jsonl \
#     20 /tmp/spikenaut-out train
#
# Ingest is JSONL (JSON3 + stdlib). Published v3 shards are parquet; this
# sidecar does not convert or invent timestamps. Point it at JSONL records
# that already carry the five live fields + episode_id.
#
# Do not export weights to Hugging Face. Do not write rmems/Spikenaut-SNN
# `dataset/merged_v2/`. K-WTA stays on in training. Health eval is k=none
# on test gpu-000170..198 (mean pairwise cofire + all-16 + I spikes).
# Errors if the JSONL has no test episodes — will not silently eval train.
# health_eval deep-copies the bank so export membrane stays post-train.

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
const DEFAULT_V3_JSONL = "/home/raulmc/Spikenaut-Vault/Spikenaut-SNN-Telemetry/v3/state_telemetry.jsonl"

# Legal v3 live columns → axons 0..4. Order is the encoder contract.
const LIVE_COLUMNS = (
    :mem_util_pct,
    :power_w,
    :gpu_temp_c,
    :sm_clock_mhz,
    :mem_clock_mhz,
)
const N_LIVE_AXONS = length(LIVE_COLUMNS)   # 5; axons 5..15 unused

# Frozen minmax from v3 state_telemetry train split (sha lineage 74acdd0f).
# Do not recompute on val/test. Scale then clamp to [0, 1].
const FROZEN_MINMAX = (
    mem_util_pct  = (0.0,                    75.0),
    power_w       = (8.527000427246094,      302.8450012207031),
    gpu_temp_c    = (0.0,                    69.0),
    sm_clock_mhz  = (180.0,                  2910.0),
    mem_clock_mhz = (405.0,                  14801.0),
)
const FROZEN_LINEAGE = "74acdd0f"

# Episode holdout. Session key is episode_id (ts_utc is 100% null on v3).
const TRAIN_EP_LO, TRAIN_EP_HI = 0, 138
const VAL_EP_LO,   VAL_EP_HI   = 140, 168
const TEST_EP_LO,  TEST_EP_HI  = 170, 198
const EMBARGO_EPS = (139, 169)

# Sensors that must never become axons. Named so tests can grep the refusal.
const FORBIDDEN_SENSORS = (
    :hashrate_mh_derived, :power_w_derived, :gpu_temp_c_derived,
    :reward_hint_derived, :tick_rate, :fan_speed_pct, :vddcr_gfx_v,
    :vram_temp_c, :step_idx,
)

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

"""
    init_hidden_weights() -> Matrix{Float32}

Incoming weights, `N_NEURONS × N_CHANNELS`. Every neuron — excitatory and
inhibitory alike — gets the same mildly-positive random drive, so all 16 can
reach threshold. The E/I distinction lives on the outgoing side; see
[`apply_dale_out!`](@ref).
"""
function init_hidden_weights()
    return randn(Float32, N_NEURONS, N_CHANNELS) .* 0.08f0 .+ 0.04f0
end

"""
    init_readout() -> Matrix{Float32}

Outgoing weights, `N_OUTPUTS × N_NEURONS`. Column `i` is neuron `i`'s
projection, sign-structured by Dale: excitatory columns non-negative,
inhibitory columns non-positive.
"""
function init_readout()
    R = randn(Float32, N_OUTPUTS, N_NEURONS) .* 0.05f0
    @inbounds for o in 1:N_OUTPUTS
        for i in 1:N_EXC
            R[o, i] = abs(R[o, i])
        end
        for i in INHIB_ROWS
            R[o, i] = -abs(R[o, i])
        end
    end
    return R
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
        init_readout(),
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

"""
    rec_has(sample, names...) -> Bool

True when any named **key** is present, including a JSON `null` value.
`rec_get` cannot be used for this: a present null returns `nothing` and
looks like a missing key.
"""
function rec_has(sample, names::Symbol...)
    for name in names
        haskey(sample, name) && return true
        haskey(sample, String(name)) && return true
    end
    return false
end

function rec_f32(sample, default::Float32, names::Symbol...)
    v = rec_get(sample, names...)
    v === nothing && return default
    return Float32(v)
end

# ── Stimulus / reward from v3 state_telemetry or legacy spike rows ────
"""
    frozen_unit01(x, lo, hi) -> Float32

`(x - lo) / (hi - lo)` then clamp to `[0, 1]`. `nothing` / missing is **0**,
not an imputed neighbour or a refit minmax. T=0 stays 0.
"""
function frozen_unit01(x, lo::Real, hi::Real)
    x === nothing && return 0f0
    span = Float32(hi) - Float32(lo)
    span == 0f0 && return 0f0
    return clamp((Float32(x) - Float32(lo)) / span, 0f0, 1f0)
end

function frozen_unit01(col::Symbol, x)
    lo, hi = getfield(FROZEN_MINMAX, col)
    return frozen_unit01(x, lo, hi)
end

"""
    live_f32(sample, col) -> Union{Float32, Nothing}

Read one legal live column. Does **not** fall back to `*_derived`.
JSON `null` and a missing key are both `nothing` (encode as 0).
"""
function live_f32(sample, col::Symbol)
    v = rec_get(sample, col)
    v === nothing && return nothing
    return Float32(v)
end

"""
    is_state_telemetry(sample) -> Bool

True when the record has any of the five legal live **keys**.
A JSON `null` still counts — the value encodes as 0 (T=0 stays 0).
`*_derived` / `tick_rate` do **not** count — those are forbidden sensors.
"""
function is_state_telemetry(sample)
    rec_has(sample, LIVE_COLUMNS...)
end

"""
    is_forbidden_derived(sample) -> Bool

True for `qubic_ticks_snn` rows that only have `tick_rate` + `*_derived`.
v3 `state_telemetry` does not contain those field names.
"""
function is_forbidden_derived(sample)
    rec_get(sample, :reward_hint_derived, :hashrate_mh_derived,
            :gpu_temp_c_derived, :power_w_derived, :tick_rate) !== nothing
end

"""
    to_stimuli(sample, enc=nothing) -> Vector{Float32}

16-channel Poisson rates.

- Legacy: `spikes` / `inputs` (clamped to [0, 1], padded/truncated to 16).
- v3 `state_telemetry`: axons 0..4 are the five live columns after frozen
  minmax; axons 5..15 stay 0 as unused width (not fake channels, not
  first-differences, not composites).

`enc` is accepted for call-site compatibility and ignored — there is no
history channel. Pointing this at `qubic_ticks_snn` (`*_derived`) errors
instead of silently training on a closed form of `tick_rate`.
"""
function to_stimuli(sample, enc=nothing)
    raw = rec_get(sample, :spikes, :inputs)
    if raw !== nothing
        stim = zeros(Float32, N_CHANNELS)
        n = min(length(raw), N_CHANNELS)
        for i in 1:n
            stim[i] = clamp(Float32(raw[i]), 0f0, 1f0)
        end
        return stim
    end

    if is_state_telemetry(sample)
        stim = zeros(Float32, N_CHANNELS)
        @inbounds for (i, col) in enumerate(LIVE_COLUMNS)
            stim[i] = frozen_unit01(col, live_f32(sample, col))
        end
        # axons 5..15 (Julia 6:16) remain 0 — unused width.
        return stim
    end

    is_forbidden_derived(sample) && error(
        "Refusing *_derived / tick_rate sensors (closed form of tick_rate; " *
        "Spikenaut Scientist exp-008, 0 mismatches / 27430). " *
        "Legal v3 live columns: mem_util_pct, power_w, gpu_temp_c, " *
        "sm_clock_mhz, mem_clock_mhz. Unused axons 5..15 stay 0."
    )

    error(
        "Sample is missing `spikes`/`inputs` and is not v3 state_telemetry " *
        "(expected mem_util_pct / power_w / gpu_temp_c / sm_clock_mhz / mem_clock_mhz)"
    )
end

"""
    sample_reward(sample) -> Float32

Signed learning signal in `[-1, 1]`.

v3: thermal / power pain from **live** `gpu_temp_c` and `power_w` after
the frozen scale (hot + high-power → negative). Never reads `*_derived`.

Legacy spike rows may still carry `reward` / `target_reward` / `reward_hint`.
"""
function sample_reward(sample)::Float32
    if is_state_telemetry(sample)
        temp_u  = frozen_unit01(:gpu_temp_c, live_f32(sample, :gpu_temp_c))
        power_u = frozen_unit01(:power_w, live_f32(sample, :power_w))
        return clamp(1f0 - temp_u - 0.5f0 * power_u, -1f0, 1f0)
    end
    hint = rec_f32(sample, 0.5f0, :reward, :target_reward, :reward_hint)
    return clamp(2f0 * (hint - 0.5f0), -1f0, 1f0)
end

function sample_readout_target(sample)::Vector{Float32}
    if is_state_telemetry(sample)
        temp_u  = frozen_unit01(:gpu_temp_c, live_f32(sample, :gpu_temp_c))
        power_u = frozen_unit01(:power_w, live_f32(sample, :power_w))
        # Three heads stay (comfort, temp, power). Comfort is 1 minus the
        # same live pains the reward uses — not a fabricated hint, not
        # reward_hint_derived.
        comfort = clamp(1f0 - 0.5f0 * temp_u - 0.5f0 * power_u, 0f0, 1f0)
        return Float32[comfort, temp_u, power_u]
    end
    hint = rec_f32(sample, 0.5f0, :reward, :target_reward, :reward_hint)
    return Float32[clamp(hint, 0f0, 1f0), 0f0, 0f0]
end

# ── Episode holdout ───────────────────────────────────────────────────
"""
    episode_index(id) -> Union{Int, Nothing}

Parse `gpu-000138` → 138. Does not invent an index from row order or time.
"""
function episode_index(id)
    id === nothing && return nothing
    s = String(id)
    m = match(r"gpu-(\d+)$", s)
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function episode_index_of(sample)
    return episode_index(rec_get(sample, :episode_id))
end

"""
    episode_split(id) -> Union{Symbol, Nothing}

`:train` / `:val` / `:test`, or `nothing` for embargo / unparseable.
"""
function episode_split(id)
    e = id isa Integer ? id : episode_index(id)
    e === nothing && return nothing
    e in EMBARGO_EPS && return nothing
    TRAIN_EP_LO <= e <= TRAIN_EP_HI && return :train
    VAL_EP_LO   <= e <= VAL_EP_HI   && return :val
    TEST_EP_LO  <= e <= TEST_EP_HI  && return :test
    return nothing
end

function parse_split(name::AbstractString)
    n = lowercase(strip(name))
    n in ("train", "tr") && return :train
    n in ("val", "validation", "valid") && return :val
    n in ("test", "te") && return :test
    error("Unknown split '$name' (expected train|val|test)")
end

"""
    require_train_cli_split(split) -> :train

CLI may only train on `:train`. `:val` / `:test` are holdout — refusing
`tick!(learn=true)` on them. Health eval is always k=none on test via
[`require_test_split`](@ref), independent of this argv.
"""
function require_train_cli_split(split::Symbol)
    split === :train || error(
        "CLI split must be train (got $split). " *
        "val/test are holdout; refusing tick!(learn=true) on them. " *
        "Health eval stays k=none on test via require_test_split."
    )
    return split
end

"""
    filter_split(samples, split) -> Vector

Keep rows whose `episode_id` belongs to `split`. Embargo 139 and 169 drop.
Order is the file order — never shuffled. Rows without a parseable
`episode_id` are dropped (not assigned by position).
"""
function filter_split(samples, split::Symbol)
    out = empty(samples)
    for sample in samples
        episode_split(episode_index_of(sample)) === split && push!(out, sample)
    end
    return out
end

"""
    require_test_split(samples) -> Vector

Health acceptance is k=none on test `gpu-000170..198` (exp-009 / exp-013).
Returns those rows in file order. Errors if `episode_id` is missing or if
no test episodes exist — never falls back to the train split.
"""
function require_test_split(samples)
    any(s -> episode_index_of(s) !== nothing, samples) || error(
        "No episode_id on records; cannot select test gpu-000170..198. " *
        "Refusing to evaluate health on the train split."
    )
    test_samples = filter_split(samples, :test)
    isempty(test_samples) && error(
        "No test episodes (gpu-000170..198) in this JSONL. " *
        "Health eval is k=none on the test split (exp-009/013); " *
        "refusing to evaluate the train split."
    )
    return test_samples
end

function reset_temporal!(bank::LIFBank)
    fill!(bank.v, 0f0)
    fill!(bank.pre_tr, 0f0)
    fill!(bank.elig, 0f0)
    fill!(bank.spikes, false)
    return bank
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

"""
    apply_dale_out!(R) -> R

Dale's law on **outgoing** weights: column `i` of the readout holds neuron `i`'s
projections, so excitatory neurons are constrained non-negative there and
inhibitory neurons non-positive.

This used to constrain the *incoming* `bank.weights` rows instead, which is both
backwards — a neuron is excitatory or inhibitory by what it does to its targets,
not by what it receives — and fatal: with inhibitory rows clipped to ≤ 0 and
`stim` non-negative, rows 13:16 could only integrate downward and never reached
the +1 threshold. Measured over 4000 ticks they fired 0 times, membrane resting
near −30, so K-WTA could never select them and the exported "E/I" model had no
inhibition in it at all.
"""
function apply_dale_out!(R::AbstractMatrix)
    @inbounds for o in 1:size(R, 1)
        for i in 1:N_EXC
            R[o, i] < 0 && (R[o, i] = 0f0)
        end
        for i in INHIB_ROWS
            R[o, i] > 0 && (R[o, i] = 0f0)
        end
    end
    return R
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
"""
    tick!(bank, stim, reward, target=nothing; k=K_WTA)

`k=nothing` is health eval (k=none): every neuron that crosses threshold
stays a spike. Training keeps `k=K_WTA`.
"""
function tick!(bank::LIFBank, stim::Vector{Float32}, reward::Float32,
               target::Union{Vector{Float32},Nothing}=nothing;
               k::Union{Int,Nothing}=K_WTA, learn::Bool=true)
    # 1. Poisson pre-spikes
    pre = Float32.(rand(Float32, N_CHANNELS) .< stim)

    # 2. OTTT presynaptic trace
    bank.pre_tr .= TRACE_LAMBDA .* bank.pre_tr .+ pre
    pre_snap = copy(bank.pre_tr)

    # 3. LIF forward: decay is a **keep** factor (Rust leak = 1 - keep).
    input = bank.weights * stim
    bank.v .= bank.decay .* bank.v .+ input

    # 4. Fire, then optional K-WTA, then reset winners only
    bank.spikes .= bank.v .>= bank.thresh
    if k !== nothing
        apply_kwta!(bank.spikes, bank.v, k)
    end
    bank.v[bank.spikes] .= 0f0

    if learn
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

        # 7. 16×3 readout: predict (comfort, temp, power) from this tick's spikes.
        #    Plain supervised delta rule — NOT reward-modulated. `reward` is signed
        #    and goes negative on exactly the thermal/power rows this readout must
        #    predict, so scaling by it ascends the error and trains the map
        #    backwards. Reward modulation belongs on the e-prop path above (step 6).
        if target !== nothing
            s = Float32.(bank.spikes)
            pred = bank.readout * s
            err = target .- pred
            bank.readout .+= READOUT_LR .* (err * s')
            apply_dale_out!(bank.readout)
        end

        # 8. Signed L2 cap on incoming weights (do **not** divide rows by sum — that
        #    destroyed sign and collapsed every row onto the same simplex). Incoming
        #    weights carry no E/I sign constraint: Dale lives on the outgoing side,
        #    in step 7, so every neuron can be driven to threshold.
        scale_rows_l2!(bank.weights, ROW_L2_CAP)
        clamp!(bank.weights, W_MIN, W_MAX)
    end

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

function _looks_like_parquet(path)
    isfile(path) && endswith(lowercase(path), ".parquet") && return true
    isdir(path) || return false
    for entry in readdir(path)
        endswith(lowercase(entry), ".parquet") && return true
    end
    return false
end

function load_data(data_path)
    # Published v3 `state_telemetry` is parquet shards. This sidecar stays
    # JSON3 + stdlib: it reads record-shaped JSONL with the five live fields.
    # A converter is not bundled and timestamps are not invented.
    if _looks_like_parquet(data_path)
        error(
            "Refusing parquet at $data_path. " *
            "Ingest is JSONL records with mem_util_pct, power_w, gpu_temp_c, " *
            "sm_clock_mhz, mem_clock_mhz, episode_id. " *
            "Do not invent ts_utc. Do not silently train on *_derived."
        )
    end
    if isdir(data_path)
        load_chunked_dir(data_path)
    elseif isfile(data_path)
        load_jsonl(data_path)
    else
        error("Data path not found: $data_path")
    end
end

"""
    assert_legal_sensors!(samples)

Fail if the corpus is `*_derived` / `tick_rate` with no live columns, so
pointing the default path at qubic_ticks_snn cannot silently train on
forbidden sensors. Mixed rows that already have the five live fields are
allowed — `to_stimuli` ignores `*_derived` on those.
"""
function assert_legal_sensors!(samples)
    n_live = count(is_state_telemetry, samples)
    n_derived = count(s -> is_forbidden_derived(s) && !is_state_telemetry(s), samples)
    if n_live == 0 && n_derived > 0
        error(
            "All $n_derived telemetry rows are *_derived / tick_rate " *
            "(forbidden; exp-008). Pass v3 state_telemetry JSONL with " *
            "mem_util_pct, power_w, gpu_temp_c, sm_clock_mhz, mem_clock_mhz."
        )
    end
    return samples
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
            # E/I sign lives on each neuron's outgoing projection
            # (`output_weights`), not on its incoming `weights` row.
            "dale"           => "outgoing",
            "k_wta"          => K_WTA,
            "q88"            => "signed",
            "decay_semantics"=> "keep",
            "n_outputs"      => N_OUTPUTS,
            "encoder"        => "v3_state_telemetry",
            "legal_columns"  => collect(string.(LIVE_COLUMNS)),
            "unused_axons"   => "5:15",
            "frozen_minmax"  => Dict(string(k) => [lo, hi] for (k, (lo, hi)) in pairs(FROZEN_MINMAX)),
            "frozen_lineage" => FROZEN_LINEAGE,
            "episode_split"  => Dict(
                "train"   => "gpu-000000..138",
                "val"     => "gpu-000140..168",
                "test"    => "gpu-000170..198",
                "embargo" => [139, 169],
            ),
        ))
    end

    write_mem(joinpath(out_dir, "parameters.mem"), bank.thresh)
    # NOTE the chained `for i ... for ch ...`, not the comma form. A comma
    # generator is a cartesian product and iterates column-major (i fastest),
    # which would silently reorder these memories; the nested `for` loop this
    # replaced varies the *last* index fastest. Verified byte-identical.
    write_mem(joinpath(out_dir, "parameters_weights.mem"),
              (bank.weights[i, ch] for i in 1:N_NEURONS for ch in 1:N_CHANNELS))
    write_mem(joinpath(out_dir, "parameters_decay.mem"), bank.decay)
    # 48 signed values, neuron-major: 16 neurons × 3 readout heads
    write_mem(joinpath(out_dir, "parameters_output_weights.mem"),
              (bank.readout[o, i] for i in 1:N_NEURONS for o in 1:N_OUTPUTS))

    return (
        joinpath(out_dir, "snn_model.json"),
        joinpath(out_dir, "parameters.mem"),
        joinpath(out_dir, "parameters_weights.mem"),
        joinpath(out_dir, "parameters_decay.mem"),
        joinpath(out_dir, "parameters_output_weights.mem"),
    )
end

"""
    mean_pairwise_cofire(spike_counts, both) -> Float64

Mean pairwise cosine of spike trains: for each pair of neurons that
fired at least once, `both_ij / sqrt(n_i * n_j)`. Silent neurons are
excluded so 11 lockstep + 5 silent → ~1 (not 55/120). Matches Scientist
"mean pairwise cofire" (exp-009: k=4 from 16 → 0.20; all-16 → 1.0).
"""
function mean_pairwise_cofire(spike_counts, both)
    acc = 0.0
    np = 0
    n = length(spike_counts)
    @inbounds for i in 1:(n - 1)
        ni = spike_counts[i]
        ni == 0 && continue
        for j in (i + 1):n
            nj = spike_counts[j]
            nj == 0 && continue
            acc += both[i, j] / sqrt(Float64(ni) * Float64(nj))
            np += 1
        end
    end
    return np == 0 ? 0.0 : acc / np
end

"""
    health_eval(bank, samples) -> NamedTuple

Forward pass with k=none (no K-WTA). Training may use k=4; health does not.
Resets membrane at each `episode_id` boundary. Does not update weights.

Deep-copies `bank` so `snn_model.json` `membrane_potential` stays
**post-train**, not post-test. Weights are already safe (`learn=false`).

Call with the **test** split (`gpu-000170..198`). The CLI default train
filter must not be passed here — train health leaks and cannot close the
exp-009 bar (cofire 0.891 / all-16 0.311).
"""
function health_eval(bank::LIFBank, samples)
    bank = deepcopy(bank)
    reset_temporal!(bank)
    total_spikes = 0
    all16 = 0
    inhib_spikes = 0
    n_pat = Set{UInt32}()
    spike_counts = zeros(Int, N_NEURONS)
    both = zeros(Int, N_NEURONS, N_NEURONS)
    prev_ep = nothing
    for sample in samples
        ep = episode_index_of(sample)
        if ep !== prev_ep
            reset_temporal!(bank)
            prev_ep = ep
        end
        nspk = tick!(bank, to_stimuli(sample), 0f0, nothing; k=nothing, learn=false)
        total_spikes += nspk
        nspk == N_NEURONS && (all16 += 1)
        inhib_spikes += count(bank.spikes[INHIB_ROWS])
        bits = UInt32(0)
        fired = Int[]
        @inbounds for i in 1:N_NEURONS
            if bank.spikes[i]
                bits |= UInt32(1) << (i - 1)
                spike_counts[i] += 1
                push!(fired, i)
            end
        end
        @inbounds for a in 1:(length(fired) - 1)
            i = fired[a]
            for b in (a + 1):length(fired)
                both[i, fired[b]] += 1
            end
        end
        push!(n_pat, bits)
    end
    n = max(length(samples), 1)
    return (
        n = length(samples),
        spk_per_tick = total_spikes / n,
        all16_frac = all16 / n,
        cofire = mean_pairwise_cofire(spike_counts, both),
        patterns = length(n_pat),
        inhib_spikes = inhib_spikes,
        k = "none",
    )
end

# ── Main ──────────────────────────────────────────────────────────────
function main(args=ARGS)
    length(args) >= 1 || error(
        "Usage: julia scripts/spikenaut_train.jl <data_path> [epochs] [out_dir] [split]\n" *
        "  data_path: v3 state_telemetry JSONL (5 live columns + episode_id).\n" *
        "  live:      mem_util_pct, power_w, gpu_temp_c, sm_clock_mhz, mem_clock_mhz\n" *
        "  frozen:    mem_util 0..75; power 8.527..302.845; temp 0..69;\n" *
        "             sm 180..2910; memclk 405..14801  (train, lineage $FROZEN_LINEAGE)\n" *
        "  holdout:   train gpu-000000..138; val 140..168; test 170..198;\n" *
        "             embargo 139 and 169. Never shuffle across episodes.\n" *
        "  unused:    axons 5..15 held at 0 (width, not fake channels).\n" *
        "  health:    k=none on test gpu-000170..198 (cofire, all-16, I spikes).\n" *
        "             Errors if the JSONL has no test episodes.\n" *
        "  split:     train only (default). val/test error — no learn=true on holdout.\n" *
        "  example:   julia scripts/spikenaut_train.jl $DEFAULT_V3_JSONL 20 /tmp/spikenaut-out train"
    )
    data_path = args[1]
    epochs    = length(args) >= 2 ? parse(Int, args[2]) : 20
    out_dir   = length(args) >= 3 ? args[3] : "out_train"
    split     = require_train_cli_split(length(args) >= 4 ? parse_split(args[4]) : :train)

    isdir(data_path) || isfile(data_path) || error("Data path not found: $data_path")
    mkpath(out_dir)

    println("=== Spikenaut Julia Trainer ===")
    println("Data   : $data_path")
    println("Epochs : $epochs")
    println("Out    : $out_dir")
    println("Split  : $split  (train gpu-000000..138 / val 140..168 / test 170..198; embargo 139,169)")
    println("Live   : mem_util_pct, power_w, gpu_temp_c, sm_clock_mhz, mem_clock_mhz")
    println("Scale  : frozen train minmax lineage=$FROZEN_LINEAGE; axons 5..15 unused=0")
    println("Dale   : $N_EXC excitatory / $N_INHIB inhibitory (outgoing readout); incoming W unsigned")
    println("K-WTA  : train k=$K_WTA; health eval k=none on test gpu-000170..198")
    println("Decay  : keep=$DECAY  (Rust leak = $(1 - DECAY))")

    print("Loading samples... ")
    loaded = load_data(data_path)
    println("$(length(loaded)) total records")
    isempty(loaded) && error("No valid samples found.")
    assert_legal_sensors!(loaded)

    has_episodes = any(s -> episode_index_of(s) !== nothing, loaded)
    samples = if has_episodes
        filter_split(loaded, split)
    else
        collect(loaded)
    end
    if has_episodes
        println("Holdout : $(length(samples)) $split rows (file order, no shuffle)")
        isempty(samples) && error("No rows in split $split after embargo filter.")
    elseif any(is_state_telemetry, loaded)
        error("v3 state_telemetry rows need episode_id for the session holdout. " *
              "Refusing to invent a split or a timestamp.")
    end
    if length(samples) < MIN_TRAIN_N
        @warn "Only $(length(samples)) records — monotonic / collapsed weights are likely."
    end

    # Health is always the held-out test split. Resolve it before training so a
    # train-only JSONL errors instead of silently printing train health.
    test_samples = nothing
    if any(is_state_telemetry, loaded)
        test_samples = require_test_split(loaded)
        println("Health  : k=none on test gpu-000170..198 ($(length(test_samples)) rows); not the $split split")
    end

    bank = LIFBank()

    for epoch in 1:epochs
        total_reward = 0f0
        total_spikes = 0
        max_spikes   = 0

        # Every piece of per-tick temporal state resets together. Clearing only
        # `v` left `pre_tr`/`elig`/`spikes` carrying the tail of the previous
        # replay, so epoch 2+ opened with stale eligibility.
        reset_temporal!(bank)
        prev_ep = nothing

        t0 = time()
        for sample in samples
            ep = episode_index_of(sample)
            if ep !== prev_ep
                # Membrane reset at episode_id (exp-010). Do not cross sessions.
                reset_temporal!(bank)
                prev_ep = ep
            end
            stim   = to_stimuli(sample)
            reward = sample_reward(sample)
            target = sample_readout_target(sample)
            nspk   = tick!(bank, stim, reward, target; k=K_WTA)
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
        n_inh  = count(i -> all(<=(0), view(bank.readout, :, i)), INHIB_ROWS)
        ms_tick = elapsed * 1000 / n

        @printf("Epoch %3d/%d | reward=%+.4f | spike_rate=%.3f | max_spk=%d | w=%+.4f±%.4f [%+.3f,%+.3f] | inhib_cols=%d | %.3fms/tick\n",
                epoch, epochs, avg_r, s_rate, max_spikes, w_mean, w_std, w_min, w_max, n_inh, ms_tick)
    end

    if test_samples !== nothing
        h = health_eval(bank, test_samples)
        @printf("Health k=none | split=test gpu-000170..198 | n=%d | spk/tick=%.3f | all-16=%.3f | cofire=%.3f | patterns=%d | I_spikes=%d\n",
                h.n, h.spk_per_tick, h.all16_frac, h.cofire, h.patterns, h.inhib_spikes)
    end

    paths = export_artifacts(bank, out_dir)
    println("\nExported:")
    for p in paths
        println("  $p")
    end
    println("Hidden weights: min=$(minimum(bank.weights)) max=$(maximum(bank.weights)) std=$(std(bank.weights))")
    println("SUCCESS: Spikenaut trained (v3 live encoder + outgoing Dale + K-WTA + signed Q8.8).")
    return bank
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
