# Changelog

## Unreleased

### Changed

- `scripts/spikenaut_train.jl` / README: reconcile leftover Spikenaut-SNN#13 surfaces after #34. Incoming `W` banner no longer says “unsigned” (it is signed-capable with no Dale lock; Q8.8 stays two's-complement). Export now decodes written `.mem` files (`q88_decode` + `assert_signed_export`) so mixed-sign hidden + Dale 80:20 cannot pass by matching an unsigned encoder to itself. README records this script as the `merged_v2` *replacement source* without writing that tree or publishing HF.

- Post-transfer hygiene (#27): live docs and package metadata point at [`rmems/SynapticDistill.jl`](https://github.com/rmems/SynapticDistill.jl). README / AGENTS record that the GitHub wiki is enabled under rmems. The plasticity-lab boundary still links the live external peer [`Limen-Neural/plasticity-lab`](https://github.com/Limen-Neural/plasticity-lab) (`rmems/plasticity-lab` does not exist).

- `scripts/spikenaut_train.jl`: anti-clone / I-drive knobs from Spikenaut Scientist **exp-023** (seed 123 / 5 ep PASS: cofire 0.733, I live, 10 unique active Q8.8 of 12 active). `DIV_LR=0.00035`, `DIV_COS_MIN=0.55`, `I_DRIVE=0.05`, `I_THRESH=0.90`, `I_WTA_MAX=2`, `E_WTA_MIN=2`, `STDP_LTD=0.0008`, `RATE_TARGET=0.12`. LIVE_COLUMNS / FROZEN_MINMAX / episode holdout / health_eval k=none unchanged. Optional CLI seed (default 123). Does not write `merged_v2` or publish HF.

- `scripts/spikenaut_train.jl`: default encoder is legal v3 `state_telemetry` (five live columns: `mem_util_pct`, `power_w`, `gpu_temp_c`, `sm_clock_mhz`, `mem_clock_mhz`) with frozen train minmax (sha lineage `74acdd0f`) and `episode_id` holdout (`gpu-000000..138` / `140..168` / `170..198`, embargo 139 and 169). Axons 5..15 stay 0 as unused width. `*_derived` / `tick_rate` are refused — they are a closed form of `tick_rate` (Scientist exp-008). Reward and readout use live `power_w` and `gpu_temp_c`. Outgoing Dale and unsigned incoming `W` unchanged. K-WTA stays on in training; health eval is `k=none` on test `gpu-000170..198` (cofire + all-16 + I spikes). JSONL ingest only; parquet is not silently converted and timestamps are not invented.

Cite: **Spikenaut Scientist** · exp-008..023

### Fixed

- `scripts/spikenaut_train.jl`: `filter_split` errors on a missing or malformed v3 `episode_id` instead of silently dropping the row. Six-digit `gpu-######` contract, embargo 139/169 drop, contiguous-episode guard, and no-shuffle file order are unchanged.
- `scripts/spikenaut_train.jl` holdout integrity (Scientist exp-014): `health_eval` deep-copies the bank so `snn_model.json` `membrane_potential` stays post-train; CLI `val`/`test` error instead of `tick!(learn=true)` on the holdout; `is_state_telemetry` detects live **key** presence so a JSON `null` on `mem_util_pct` still encodes as 0.
- `scripts/spikenaut_train.jl`: health eval no longer runs on the CLI/train filter. After training on `gpu-000000..138`, `health_eval` is k=none on test `gpu-000170..198` and prints mean pairwise cofire (exp-009 bar 0.891 / all-16 0.311). A JSONL with no test episodes errors instead of silently evaluating train (Scientist exp-013).
- `scripts/spikenaut_train.jl`: signed E/I (Dale 80:20), K-WTA, STDP depression, signed reward, 16×3 readout, and signed two's-complement Q8.8 (emits `FFF9`). Does not invert `bank.decay` (keep factor).
- **Dale's law applies to outgoing weights, not incoming.** Each neuron's E/I sign now constrains its `output_weights` (readout) column; `parameters_weights.mem` is sign-unconstrained. Constraining the incoming rows kept every inhibitory neuron below threshold forever — measured at 0 spikes in 4000 ticks, membrane resting near −30 against a +1 threshold — so the exported "E/I" model contained no inhibition. `snn_model.json` now carries `"dale": "outgoing"`.

Cite: **Grok Build: Grok 4.6 (high)**
