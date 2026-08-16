# Changelog

## Unreleased

### Fixed

- `scripts/spikenaut_train.jl`: signed E/I (Dale 80:20), K-WTA, STDP depression, signed reward (including `reward_hint_derived`), 16×3 readout, and signed two's-complement Q8.8 (emits `FFF9`). Encodes the 27k `qubic_ticks_snn.jsonl` telemetry schema instead of crashing on missing `spikes`/`inputs`. Does not invert `bank.decay` (keep factor).
- **Dale's law applies to outgoing weights, not incoming.** Each neuron's E/I sign now constrains its `output_weights` (readout) column; `parameters_weights.mem` is sign-unconstrained. Constraining the incoming rows kept every inhibitory neuron below threshold forever — measured at 0 spikes in 4000 ticks, membrane resting near −30 against a +1 threshold — so the exported "E/I" model contained no inhibition. `snn_model.json` now carries `"dale": "outgoing"`.

Cite: **Grok Build: Grok 4.6 (high)**
