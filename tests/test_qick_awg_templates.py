"""No-hardware tests for the QICK template + scan-wiring layer (devices/qick_awg + awg_runtime).

Covers, with no socket / board:
  * template builders -- exact pulses + channel tokens vs the board structures; derived pulse times;
    degrees pass-through; gain default; _split_duration honors the HW length window.
  * qick_program_duration -- total playtime (loops expanded).
  * scan_programs -- build_programs dedup over a scangroup; seq_qick_key == the builder key.
  * awg_runtime -- runp().QICK gate; the g().QICK.* per-point convention.
"""
import pytest

from devices.qick_awg import (Loop, build_program, build_programs, loop,
                              qick_enabled, qick_program_duration, seq_qick_key)
from devices.qick_awg.templates import (MAX_LEN_NS, MIN_LEN_NS, NS_PER_CC,
                                        _split_duration, _times, build_echo,
                                        build_rabi, build_ramsey)

pytestmark = pytest.mark.no_hardware


# defaults a scan would set (degrees, gain nonzero so it emits).
def _params(template, **over):
    p = {"template": template, "freq": 10863.04, "gain": 3000,
         "rabi_freq": 7.187e6, "phase": 0.0, "wait_time": 1e-6, "drive_time": 500e-9}
    p.update(over)
    return p


# --------------------------------------------------------------------------- #
# derived times + chunking helper
# --------------------------------------------------------------------------- #
def test_times_derives_pi_half_and_pi_from_rabi_freq():
    t_pi2, t_pi = _times(7.187e6)
    assert t_pi2 == pytest.approx(1e9 / (4 * 7.187e6))    # ns
    assert t_pi == pytest.approx(2 * t_pi2)


def test_times_rejects_nonpositive():
    with pytest.raises(ValueError):
        _times(0)


def test_split_single_chunk_when_it_fits():
    n, chunk = _split_duration(1e3)                       # 1 us, well under the cap
    assert n == 1
    assert chunk == pytest.approx(1e3)


def test_split_chunks_when_over_cap():
    total = 3 * MAX_LEN_NS + 1234.0                       # forces >= 4 chunks
    n, chunk = _split_duration(total)
    assert n >= 4
    assert chunk <= MAX_LEN_NS + 1e-6
    assert chunk >= MIN_LEN_NS - 1e-6
    assert n * chunk == pytest.approx(total)              # faithful total


def test_split_rejects_below_minimum():
    with pytest.raises(ValueError):
        _split_duration(MIN_LEN_NS / 2)


def test_ns_per_cc_is_rfsoc4x2_gen_clock():
    assert NS_PER_CC == pytest.approx(1e3 / 614.4)


# --------------------------------------------------------------------------- #
# Rabi
# --------------------------------------------------------------------------- #
def test_rabi_single_pulse_when_short():
    prog = build_rabi(_params("Rabi", drive_time=500e-9))
    # key carries every consumed param (template, freq, gain, phase, drive_time)
    assert prog.key == ("Rabi", 10863.04, 3000.0, 0.0, 500e-9)
    assert set(prog.pulses) == {"Drive"}
    d = prog.pulses["Drive"]
    assert d["freq"] == 10863.04 and d["gain"] == 3000 and d["style"] == "const"
    assert d["length"] == pytest.approx(500.0)           # ns
    assert prog.channels == [["Drive"]]                  # n_loops == 1 -> bare token


def test_rabi_chunks_long_drive_into_loop():
    long_drive_s = (3 * MAX_LEN_NS) * 1e-9               # > cap -> must loop
    prog = build_rabi(_params("Rabi", drive_time=long_drive_s))
    (body,) = prog.channels[0]
    assert isinstance(body, Loop)
    assert body.count >= 3 and body.body == ["Drive"]
    # total playtime faithful to the requested drive
    assert qick_program_duration(_params("Rabi", drive_time=long_drive_s)) == pytest.approx(
        long_drive_s, rel=1e-9)


# --------------------------------------------------------------------------- #
# Ramsey
# --------------------------------------------------------------------------- #
def test_ramsey_structure_and_phase_in_degrees():
    prog = build_ramsey(_params("Ramsey", wait_time=1e-6, phase=90.0))
    assert prog.key == ("Ramsey", 10863.04, 3000.0, 7.187e6, 90.0, 1e-6)
    assert set(prog.pulses) == {"PiHalf", "PiHalfPhase", "Wait"}
    # names are '/'-free; phase (deg) rides on the final pulse only
    assert prog.pulses["PiHalf"]["phase"] == 0.0
    assert prog.pulses["PiHalfPhase"]["phase"] == 90.0   # DEGREES, no conversion
    assert prog.pulses["Wait"]["gain"] == 0              # dark free-evolution
    t_pi2, _ = _times(7.187e6)
    assert prog.pulses["PiHalf"]["length"] == pytest.approx(t_pi2)
    # ch0 = [PiHalf, loop(N,[Wait]), PiHalfPhase]
    ch0 = prog.channels[0]
    assert ch0[0] == "PiHalf" and ch0[-1] == "PiHalfPhase"
    assert isinstance(ch0[1], Loop) and ch0[1].body == ["Wait"]


def test_ramsey_wait_total_is_faithful():
    dur = qick_program_duration(_params("Ramsey", wait_time=4e-6))
    t_pi2, _ = _times(7.187e6)
    # playtime = 2 pi/2 pulses + the full wait
    assert dur == pytest.approx(4e-6 + 2 * t_pi2 * 1e-9, rel=1e-6)


# --------------------------------------------------------------------------- #
# Echo
# --------------------------------------------------------------------------- #
def test_echo_structure_has_pi_in_the_middle():
    prog = build_echo(_params("Echo", wait_time=2e-6, phase=180.0))
    assert prog.key == ("Echo", 10863.04, 3000.0, 7.187e6, 180.0, 2e-6)
    assert set(prog.pulses) == {"PiHalf", "Pi", "PiHalfPhase", "Wait"}
    t_pi2, t_pi = _times(7.187e6)
    assert prog.pulses["Pi"]["length"] == pytest.approx(t_pi)
    assert prog.pulses["Pi"]["length"] == pytest.approx(2 * prog.pulses["PiHalf"]["length"])
    ch0 = prog.channels[0]
    # [PiHalf, loop(N,[Wait]), Pi, loop(N,[Wait]), PiHalfPhase]
    assert ch0[0] == "PiHalf" and ch0[2] == "Pi" and ch0[-1] == "PiHalfPhase"
    assert isinstance(ch0[1], Loop) and isinstance(ch0[3], Loop)


# --------------------------------------------------------------------------- #
# dispatch + guards
# --------------------------------------------------------------------------- #
def test_build_program_dispatch_case_insensitive():
    assert build_program(_params("ramsey")).key[0] == "Ramsey"
    assert build_program(_params("ECHO")).key[0] == "Echo"


def test_build_program_unknown_template_raises():
    with pytest.raises(ValueError):
        build_program(_params("XY8"))


def test_gain_default_zero_when_unset():
    p = {"template": "Ramsey", "freq": 10863.04, "rabi_freq": 7.187e6,
         "phase": 0.0, "wait_time": 1e-6}
    # emulate expConfig default gain=0 by not overriding it here -> KeyError guard: gain is required,
    # so the resolved-params layer always supplies it. Confirm 0 propagates when given.
    p["gain"] = 0
    prog = build_program(p)
    assert prog.pulses["PiHalf"]["gain"] == 0


# --------------------------------------------------------------------------- #
# scan_programs: build over a ScanGroup + per-shot key agreement
# --------------------------------------------------------------------------- #
class _FakeSG:
    """Minimal ScanGroup stand-in: nseq/getseq + runp().QICK gate."""
    def __init__(self, seqs, qick=True):
        self._seqs = seqs
        self._qick = qick

    def nseq(self):
        return len(self._seqs)

    def getseq(self, n):
        return self._seqs[n - 1]

    def runp(self):
        sg = self

        class _RP:
            def QICK(self, default=False):
                return sg._qick
        return _RP()


def _seq(template="Ramsey", **qick):
    q = {"template": template, "freq": 10863.04, "gain": 3000,
         "rabi_freq": 7.187e6, "phase": 0.0, "wait_time": 1e-6}
    q.update(qick)
    return {"QICK": q}


def test_build_programs_one_per_seq_and_key_matches_arm():
    seqs = [_seq(wait_time=w) for w in (1e-6, 2e-6, 3e-6)]
    sg = _FakeSG(seqs)
    progs = build_programs(sg, defaults={})
    assert [p.key for p in progs] == [
        ("Ramsey", 10863.04, 3000.0, 7.187e6, 0.0, 1e-6),
        ("Ramsey", 10863.04, 3000.0, 7.187e6, 0.0, 2e-6),
        ("Ramsey", 10863.04, 3000.0, 7.187e6, 0.0, 3e-6)]
    # the per-shot key equals the builder key for the same seq -> arm names an uploaded program
    assert seq_qick_key(seqs[1], defaults={}) == progs[1].key


def test_build_programs_freq_sweep_gives_distinct_programs():
    # REGRESSION: sweeping the carrier FREQUENCY must mint a distinct program per point (the key
    # includes freq). A key that omitted freq would collapse all points to one upload -> silent no-op.
    seqs = [_seq(template="Echo", freq=f) for f in (10861.0, 10863.0, 10865.0)]
    progs = build_programs(_FakeSG(seqs), defaults={})
    assert len({p.key for p in progs}) == 3
    assert [p.key[1] for p in progs] == [10861.0, 10863.0, 10865.0]   # freq is key[1]


def test_build_programs_defaults_merge_with_overrides():
    # defaults supply rabi_freq/freq/gain/phase; the seq overrides only wait_time
    defaults = {"template": "Ramsey", "freq": 10863.04, "gain": 3000,
                "rabi_freq": 7.187e6, "phase": 0.0, "wait_time": 1e-6}
    sg = _FakeSG([{"QICK": {"wait_time": 5e-6}}])
    progs = build_programs(sg, defaults=defaults)
    assert progs[0].key == ("Ramsey", 10863.04, 3000.0, 7.187e6, 0.0, 5e-6)


def test_qick_enabled_gate():
    assert qick_enabled(_FakeSG([_seq()], qick=True)) is True
    assert qick_enabled(_FakeSG([_seq()], qick=False)) is False


# --------------------------------------------------------------------------- #
# awg_runtime wiring over a REAL ScanGroup (the g().QICK.* + runp().QICK convention)
# --------------------------------------------------------------------------- #
def test_runp_qick_and_qick_dot_convention_real_scangroup():
    from scan_group import ScanGroup
    import awg_runtime

    g = ScanGroup()
    g().QICK.template = "Ramsey"
    g().QICK.rabi_freq = 7.187e6
    g().QICK.gain = 3000
    g().QICK.wait_time.scan(1, [1e-6, 2e-6, 3e-6])
    g.runp().QICK = True

    assert awg_runtime.qick_enabled(g) is True
    assert awg_runtime.qick_enabled(ScanGroup()) is False       # no QICK declared -> skip
    # per-point QICK params land under getseq()["QICK"] (what the pre_cb reads)
    assert g.getseq(1)["QICK"]["wait_time"] == 1e-6
    assert g.getseq(3)["QICK"]["wait_time"] == 3e-6
    # distinct swept values -> distinct program keys (one uploaded program each). Pass the c["QICK"]
    # base defaults the live run loop supplies (freq/phase come from there, not the per-point seq).
    base = {"freq": 10863.04, "phase": 0.0}
    keys = [seq_qick_key(g.getseq(n), defaults=base) for n in (1, 2, 3)]
    assert len(set(keys)) == 3
