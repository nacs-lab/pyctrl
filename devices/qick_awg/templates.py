"""templates.py -- QICK microwave program builders (Model A: template + swept scalars).

A scan author declares QICK behavior via ``g().QICK.*`` (see ``expConfig`` ``c["QICK"]``): a
``template`` name plus scalar params (``freq``/``gain``/``rabi_freq``/``phase``/``wait_time``/
``drive_time``). Each builder here turns a resolved param dict into ONE :class:`QickProgram`
(``key`` + ``pulses`` + ``channels``). The run loop builds one program per unique swept point,
batch-uploads them all once, and arms the active one per shot (``scan_programs`` + ``awg_runtime``).

Grounded in the three live board programs (``program_cfg/*.json``), but refined -- clean pulse names
and a single chunking helper instead of MATLAB's hard-coded ``loop(10,[Wait])`` twice:

    f_Rabi    [loop(20,[Sine])]                                    -> Rabi   (drive for T)
    f_Ramsey  [Pi2, loop(10,[Wait]), loop(10,[Wait]), Pi2_Phase]   -> Ramsey (pi/2 - T - pi/2)
    f_Dipolar [Pi2, loop(10,[Wait]), Pi, loop(10,[Wait]), Pi2_Phase] -> Echo (pi/2 - T/2 - pi - T/2 - pi/2)

Two hard facts from the FPGA_AWG source drive the design:

  * **Pulse-length cap is HARDWARE.** ``compiler._get_mode_code`` requires ``3 <= length_cc < 2**16``
    (the length rides in the low 16 bits of the pulse mode register), where ``length_cc`` is on the
    generator fabric clock (rfsoc4x2 f_fabric = 614.4 MHz -> 1 cc ~ 1.6276 ns). A long drive/wait must
    be split into ``loop(N,[chunk])`` -- :func:`_split_duration` is the ONE place this is enforced.
  * **Pulse times are DERIVED** from the Rabi frequency: ``t_pi2 = 1/(4*rabi_freq)``,
    ``t_pi = 2*t_pi2`` (:func:`_times`). Author sets the physics; lengths follow.

Phase is DEGREES (server ``deg2reg``); no rad->deg conversion here. Names are ``/``-free (the server
stores each pulse as ``<name>.json``) and get namespaced ``p000_*`` per program by the manager.

Out-of-band device: QICK output is NOT in the serialized seq byte blob, so THE ONE RULE does not apply.
"""
from .fpga_awg_manager import QickProgram
from .simple_pulse import loop, simple_pulse_cfg

# rfsoc4x2 generator fabric clock: 614.4 MHz -> one clock cycle in ns. The board converts a pulse's
# ns length to cycles on THIS clock (compiler.alloc_registers: us2cycles(..., gen_ch=0)).
NS_PER_CC = 1e3 / 614.4          # ~1.6276 ns/cc

# HW pulse-length window in clock cycles (compiler._get_mode_code: 3 <= length_cc < 2**16). We cap a
# hair below 2**16 so the board's own float->int us2cycles rounding can't tip a chunk over the edge.
MIN_LEN_CC = 3
MAX_LEN_CC = 60000               # safety margin under 65535

MIN_LEN_NS = MIN_LEN_CC * NS_PER_CC     # ~4.88 ns
MAX_LEN_NS = MAX_LEN_CC * NS_PER_CC     # ~97.66 us


def _times(rabi_freq):
    """Derive (t_pi2_ns, t_pi_ns) from the Rabi frequency (Hz): t_pi2 = 1/(4*f), t_pi = 2*t_pi2."""
    if rabi_freq <= 0:
        raise ValueError("QICK rabi_freq must be > 0 (got %r)" % (rabi_freq,))
    t_pi2_ns = 1e9 / (4.0 * rabi_freq)
    return t_pi2_ns, 2.0 * t_pi2_ns


def _check_pulse_len(length_ns, what):
    """Guard a single (non-chunked) pulse against the HW length window; raise a clear error."""
    if length_ns < MIN_LEN_NS:
        raise ValueError(
            "QICK %s length %.4g ns is below the HW minimum %.4g ns (%d cc). Increase it "
            "(e.g. raise rabi_freq's pulse or the drive time)." % (what, length_ns, MIN_LEN_NS, MIN_LEN_CC))
    if length_ns > MAX_LEN_NS:
        raise ValueError(
            "QICK %s length %.4g ns exceeds the single-pulse HW cap %.4g ns (%d cc); it must be "
            "chunked via _split_duration." % (what, length_ns, MAX_LEN_NS, MAX_LEN_CC))


def _split_duration(total_ns):
    """Split a total duration (ns) into ``(n_loops, chunk_ns)`` with each chunk inside the HW window.

    Returns the smallest ``n_loops >= 1`` such that ``chunk_ns = total_ns / n_loops`` lands in
    ``[MIN_LEN_NS, MAX_LEN_NS]``, so the program plays ``loop(n_loops, [chunk])`` for a faithful total.
    ``n_loops == 1`` when the whole duration already fits a single pulse. This is the ONLY place the
    16-bit HW length cap is enforced (mirrors why MATLAB chunked the wait into 20).

    Raises if ``total_ns`` is below one minimum-length pulse (can't be represented at all).
    """
    if total_ns < MIN_LEN_NS:
        raise ValueError(
            "QICK duration %.4g ns is below one minimum pulse (%.4g ns / %d cc); cannot represent."
            % (total_ns, MIN_LEN_NS, MIN_LEN_CC))
    # Fewest chunks whose per-chunk length is still <= the cap: ceil(total / MAX_LEN_NS).
    import math
    n_loops = max(1, math.ceil(total_ns / MAX_LEN_NS))
    chunk_ns = total_ns / n_loops
    # ceil() guarantees chunk_ns <= MAX_LEN_NS; and chunk_ns >= MIN (since total >= MIN and n_loops
    # only grows when total > MAX >> MIN). Guard anyway for paranoia.
    _check_pulse_len(chunk_ns, "wait/drive chunk")
    return n_loops, chunk_ns


def build_rabi(params):
    """Rabi: drive the carrier for ``drive_time`` (chunked). channels ch0 = ``[loop(N,[Drive])]``.

    ``key = ("Rabi", drive_time)``. The drive pulse itself carries the gain; sweeping ``drive_time``
    mints one program per value.
    """
    freq = params["freq"]
    gain = params["gain"]
    phase = params.get("phase", 0.0)
    drive_ns = float(params["drive_time"]) * 1e9
    n_loops, chunk_ns = _split_duration(drive_ns)
    pulses = {"Drive": simple_pulse_cfg("Drive", freq, gain, chunk_ns, phase=phase)}
    body = "Drive" if n_loops == 1 else loop(n_loops, ["Drive"])
    channels = [[body]]
    # KEY must include EVERY param that changes the built program, so a sweep of ANY of them (freq,
    # gain, ...) mints a distinct program -- else all points collapse to one upload/arm (silent no-op).
    key = ("Rabi", float(freq), float(gain), float(phase), float(params["drive_time"]))
    return QickProgram(key=key, pulses=pulses, channels=channels)


def build_ramsey(params):
    """Ramsey: ``[PiHalf, wait(T), PiHalfPhase]`` with the wait chunked as ``loop(N,[Wait])``.

    ``key = ("Ramsey", wait_time, phase)``. ``PiHalfPhase`` carries the final-pulse ``phase`` (deg).
    """
    freq = params["freq"]
    gain = params["gain"]
    phase = params.get("phase", 0.0)
    t_pi2, _t_pi = _times(params["rabi_freq"])
    _check_pulse_len(t_pi2, "PiHalf pulse (from rabi_freq)")
    wait_ns = float(params["wait_time"]) * 1e9
    n_loops, chunk_ns = _split_duration(wait_ns)
    pulses = {
        "PiHalf":      simple_pulse_cfg("PiHalf", freq, gain, t_pi2, phase=0.0),
        "PiHalfPhase": simple_pulse_cfg("PiHalfPhase", freq, gain, t_pi2, phase=phase),
        "Wait":        simple_pulse_cfg("Wait", freq, 0, chunk_ns, phase=0.0),   # gain 0 = dark
    }
    channels = [["PiHalf", loop(n_loops, ["Wait"]), "PiHalfPhase"]]
    # KEY covers every consumed param (see build_rabi) so a sweep of freq/gain/rabi_freq/phase/wait
    # each mints a distinct program.
    key = ("Ramsey", float(freq), float(gain), float(params["rabi_freq"]),
           float(phase), float(params["wait_time"]))
    return QickProgram(key=key, pulses=pulses, channels=channels)


def build_echo(params):
    """Echo (Hahn/dipolar): ``[PiHalf, wait(T/2), Pi, wait(T/2), PiHalfPhase]``, each half chunked.

    ``key = ("Echo", wait_time, phase)``.
    """
    freq = params["freq"]
    gain = params["gain"]
    phase = params.get("phase", 0.0)
    t_pi2, t_pi = _times(params["rabi_freq"])
    _check_pulse_len(t_pi2, "PiHalf pulse (from rabi_freq)")
    _check_pulse_len(t_pi, "Pi pulse (from rabi_freq)")
    half_ns = float(params["wait_time"]) * 1e9 / 2.0
    n_loops, chunk_ns = _split_duration(half_ns)
    pulses = {
        "PiHalf":      simple_pulse_cfg("PiHalf", freq, gain, t_pi2, phase=0.0),
        "Pi":          simple_pulse_cfg("Pi", freq, gain, t_pi, phase=0.0),
        "PiHalfPhase": simple_pulse_cfg("PiHalfPhase", freq, gain, t_pi2, phase=phase),
        "Wait":        simple_pulse_cfg("Wait", freq, 0, chunk_ns, phase=0.0),   # gain 0 = dark
    }
    channels = [["PiHalf", loop(n_loops, ["Wait"]), "Pi", loop(n_loops, ["Wait"]), "PiHalfPhase"]]
    # KEY covers every consumed param (see build_rabi) so a sweep of freq/gain/rabi_freq/phase/wait
    # each mints a distinct program.
    key = ("Echo", float(freq), float(gain), float(params["rabi_freq"]),
           float(phase), float(params["wait_time"]))
    return QickProgram(key=key, pulses=pulses, channels=channels)


# template name -> builder. Case-insensitive lookup via :func:`build_program`.
TEMPLATES = {
    "rabi":   build_rabi,
    "ramsey": build_ramsey,
    "echo":   build_echo,
}


def build_program(params):
    """Dispatch on ``params["template"]`` (case-insensitive) -> the built :class:`QickProgram`."""
    name = str(params.get("template", "")).strip().lower()
    builder = TEMPLATES.get(name)
    if builder is None:
        raise ValueError("Unknown QICK template %r; choose from %s"
                         % (params.get("template"), sorted(TEMPLATES)))
    return builder(params)


def qick_program_duration(params):
    """Total playtime of the built program, in SECONDS -- for the step's TTL-gate width.

    The step must hold the sequence for at least this long after the QICK trigger so the whole
    microwave program plays inside the gate window. (Pure sum of pulse lengths x loop counts; ignores
    the negligible inter-instruction tProc overhead.)
    """
    prog = build_program(params)
    # Sum each channel's token list, expanding loops; return the max across channels (they play in
    # parallel from t=0). Pulse length is read back from the built pulse cfgs (ns).
    lengths_ns = {name: cfg["length"] for name, cfg in prog.pulses.items()}

    def _tokens_ns(tokens):
        total = 0.0
        for tok in tokens:
            if isinstance(tok, str):
                total += lengths_ns.get(tok, 0.0)
            else:                                     # a Loop(count, body)
                total += tok.count * _tokens_ns(tok.body)
        return total

    per_ch = [_tokens_ns(ch) for ch in prog.channels]
    return (max(per_ch) if per_ch else 0.0) * 1e-9
