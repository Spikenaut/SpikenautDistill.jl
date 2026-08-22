# Changelog

## Unreleased

### Changed

- `scripts/spikenaut_train.jl`: default encoder is legal v3 `state_telemetry` (five live columns: `mem_util_pct`, `power_w`, `gpu_temp_c`, `sm_clock_mhz`, `mem_clock_mhz`) with frozen train minmax (sha lineage `74acdd0f`) and `episode_id` holdout (`gpu-000000..138` / `140..168` / `170..198`, embargo 139 and 169). Axons 5..15 stay 0 as unused width. `*_derived` / `tick_rate` are refused — they are a closed form of `tick_rate` (Scientist exp-008). Reward and readout use live `power_w` and `gpu_temp_c`. Outgoing Dale and unsigned incoming `W` unchanged. K-WTA stays on in training; health eval is `k=none`. JSONL ingest only; parquet is not silently converted and timestamps are not invented.

Cite: **Spikenaut Scientist** · exp-008..011

### Fixed

- `scripts/spikenaut_train.jl`: signed E/I (Dale 80:20), K-WTA, STDP depression, signed reward, 16×3 readout, and signed two's-complement Q8.8 (emits `FFF9`). Does not invert `bank.decay` (keep factor).
- **Dale's law applies to outgoing weights, not incoming.** Each neuron's E/I sign now constrains its `output_weights` (readout) column; `parameters_weights.mem` is sign-unconstrained. Constraining the incoming rows kept every inhibitory neuron below threshold forever — measured at 0 spikes in 4000 ticks, membrane resting near −30 against a +1 threshold — so the exported "E/I" model contained no inhibition. `snn_model.json` now carries `"dale": "outgoing"`.

Cite: **Grok Build: Grok 4.6 (high)**
