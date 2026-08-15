# Changelog

## Unreleased

### Fixed

- `scripts/spikenaut_train.jl`: signed E/I (Dale 80:20), K-WTA, STDP depression, signed reward (including `reward_hint_derived`), 16×3 readout, and signed two's-complement Q8.8 (emits `FFF9`). Encodes the 27k `qubic_ticks_snn.jsonl` telemetry schema instead of crashing on missing `spikes`/`inputs`. Does not invert `bank.decay` (keep factor).

Cite: **Grok Build: Grok 4.6 (high)**
