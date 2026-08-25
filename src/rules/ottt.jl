# SPDX-License-Identifier: MIT OR Apache-2.0

"""
Implementation of the Online Spatio-Temporal Trace Training (OTTT) rule.
"""

"""
    update_ottt!(model, spikes, loss; kwargs...)

Calculate gradients for the OTTT learning rule.

This is a placeholder implementation. A real implementation would:
- Maintain presynaptic and postsynaptic traces.
- Compute weight updates based on the correlation between these traces and a global learning signal (loss).
"""
function update_ottt!(model, spikes::SpikeBatch, loss; kwargs...)
    # As with e-prop, a real implementation would return a proper gradient structure.

    error("update_ottt!: the OTTT learning rule is not yet implemented.")
end
