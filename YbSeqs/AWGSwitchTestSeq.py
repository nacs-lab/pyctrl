r"""AWGSwitchTestSeq.py -- bench test: two AWG trigger edges 10 us apart with the C1/C2 RF
switch flipped in the gap. NO atoms / no science -- just FPGA TTLs, so you can watch the 556
AWG output on a scope and confirm C1 (rising gaussian, fwd) then C2 (falling gaussian, rev)
route correctly through the switch.

PREREQ -- arm the AWG channels FIRST (C1=rise, C2=fall, NCYC=1, DLAY=0, EXT trigger):
    pyctrl\.venv-engine-py312\Scripts\python.exe pyctrl\tmp\awg_switch_test.py ext
This seq does NOT program the AWG (the scan sets AWGs=[] so AWGManager leaves the box alone).
It only drives the FPGA trigger (TTL556RydAWG = TTL1) and the C1/C2 RF switch select
(TTL556RydAWGChSwitch = TTL16).

Scope: trigger on TTLScopeTrig (FPGA1/TTL3), ~20 us window. Expect a rising gaussian at
t~=1 us and a falling gaussian at t~=11 us.
"""
# RF switch select levels (SPDT, single control line = TTL16). If the scope shows fwd/rev
# SWAPPED, swap these two values.
SEL_C1 = 0   # route C1 -> forward -> rising gaussian
SEL_C2 = 1   # route C2 -> reverse -> falling gaussian

TRIG_SPACING_US = 5.5   # edge-to-edge spacing of the two triggers
FLIP_AT_US = 5.5         # flip the switch this long after edge 1 (mid-gap: past the C1 pulse)


def AWGSwitchTestSeq(s):
    # scope trigger marker (rising edge = t0 of the window)
    
    # route C1 for the forward pulse
    s.add('TTL556RydAWGChSwitch', SEL_C1)
    # NOTE: the engine inserts a FIXED ~27us prepend before the first timed step, so on the scope
    # edge1 lands at ~27us + this wait (additive, same every shot -- harmless constant offset;
    # relative timing after it is exact). Real seqs hide it behind their long initial steps.
    s.wait(1e-6)                                     # settle
    s.add('TTLScopeTrig', 1)
    
    # --- trigger edge 1: both AWG channels fire; switch passes C1 (rising gaussian) ---
    s.add('TTL556RydAWG', 1)
    s.wait(0.1e-6)
    s.add('TTL556RydAWG', 0)

    # mid-gap: the ~4.3 us C1 pulse is done -> flip the switch to C2
    s.wait(FLIP_AT_US * 1e-6)
    s.add('TTL556RydAWGChSwitch', SEL_C2)
    # s.wait((TRIG_SPACING_US - FLIP_AT_US - 0.1) * 1e-6)   # arrive at edge1 + 10 us

    # --- trigger edge 2: both fire; switch passes C2 (falling gaussian) ---
    s.add('TTL556RydAWG', 1)
    s.wait(0.1e-6)
    s.add('TTL556RydAWG', 0)
    s.wait(6e-6)                                     # let the C2 pulse finish

    # close the scope window + reset the switch
    s.add('TTLScopeTrig', 0)
    s.add('TTL556RydAWGChSwitch', SEL_C1)

    # dummy camera frame so the run loop's capture (NumImages=1) is satisfied (dark frame)
    s.wait(1e-6)
    s.add('TTLOrcaTrig', 1)
    s.wait(1e-6)
    s.add('TTLOrcaTrig', 0)

    return s
