"""TTL27PulseSeq.py -- bench trigger: one 50 us HIGH pulse on FPGA1/TTL27, then LOW.

NO atoms / no science -- just a single FPGA TTL output pulse. Used to emit a hardware
trigger edge (e.g. into a scope or an external device). The 60 Hz AC-line trigger is wired
externally and gates when the FPGA starts, so this seq does NOT wait on any input -- it only
drives the output line.

TTL27 has no expConfig alias; use the RAW backend address ``FPGA1/TTL27`` (bare ``TTL27`` does
not route -- every live channel is board-prefixed, cf. the Seq_Dump). It passes through
translate_channel verbatim (lib/seq_config.py), so no alias is needed.

NOTE: the engine inserts a fixed ~27 us prepend before the first timed step, so the pulse
lands ~27 us into the shot (harmless constant offset; the 50 us width itself is exact).
"""

TTL27 = "FPGA1/TTL27"
PULSE_WIDTH_S = 50e-3


def TTL27PulseSeq(s):
    s.add_step(PULSE_WIDTH_S).add(TTL27, 1)   # 50 us finite-width step: TTL27 HIGH for its duration
    s.add_step(PULSE_WIDTH_S).add(TTL27, 0)                         # zero it after the step
    return s
