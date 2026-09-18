"""nidaq_runner.py -- the one real device-driver PORT (onto ``nidaqmx``).

Port of ``matlab_new/lib/NiDAQRunner.m``, which uses MATLAB's Data Acquisition Toolbox
session (``daq.createSession('ni')``). pyctrl drives the **NI PCIe-6738** (32 AO, 16-bit,
1 MS/s, Dev1) through the ``nidaqmx`` Python package over the same NI-DAQmx runtime. The card
is externally clocked + triggered by the FPGA (PFI0 clock ~400 kHz, PFI1 start trigger), so
this module only configures the card to LISTEN, then write -> start -> wait.

⚠ **#1 silent-bug risk (no error if wrong): the transpose.** ``run_bseq`` (run_seq2.py)
hands a **sample-major** ``[nsamps, nchns]`` matrix (MATLAB ``queueOutputData`` order);
``nidaqmx`` writes **channel-major** ``[nchns, nsamps]``. :func:`_to_channel_major` does that
transpose -- the single place it happens, covered NO-HARDWARE by the shape test.

This module imports with NO ``nidaqmx`` present: every device call is funneled through the
``_build_task`` / ``_write_and_start`` / ``_wait_task`` / ``_close_task`` hooks, which lazily
import ``nidaqmx`` only when a real run happens. The cache-invalidation decision and the
transpose are pure functions, so the NO-HARDWARE tests cover them by stubbing those hooks.
The live card-listen / clock-out is NEEDS-HARDWARE (maintenance window).

Execution details a naive port misses (finding D):
  * **Session is process-GLOBAL**, reused across all shots/jobs (adding channels is ~50 ms
    each -- the reason for the cache).
  * **Cache invalidation = STRUCTURAL equality** (``isequaln`` channels + ``map_equal``
    clocks/triggers), NOT identity -- the channel list is rebuilt every ``generate()``.
  * **``cache_in_use``** is set in ``get_session`` and cleared ONLY in :meth:`wait` -- after
    an error (``run_real``'s catch does NOT call ``wait``) the next run rebuilds the session.
  * **A FINITE task must be STOPPED before re-write each shot** (MATLAB ``queueOutputData``
    re-arms implicitly): per-shot ``stop -> write -> start -> (wait later)``.
  * **AO-channel ADD ORDER must equal the reshape column order** (channel ``i`` of the data
    is AO channel ``i``).
  * ``rate = 400e3`` = the real ~400 kHz FPGA PFI0 clock. (NOT 500e3: the 6738 rejects
    >400 kHz at 14 channels -- DaqError -200332. The clock is external, so FINITE mode
    completes after ``samps_per_chan`` edges regardless; the rate only needs to be <= the
    device max and match the FPGA clock. Found on the first physical pyctrl run, 2026-06-02.)

Design inspired by the MATLAB original; no brassboard-seq code.
"""

import logging
import weakref

_log = logging.getLogger(__name__)

_RATE = 400e3          # NI sample-clock rate (Hz) = the FPGA PFI0 clock.
# ⚠ Was 500e3 ("over-estimate"), but the PCIe-6738 REJECTS >400 kHz with 14 channels
# (DaqError -200332, "Specified sample rate is higher than the fastest rate supported";
# device max = 400 kHz at 14 chn). The clock is EXTERNAL (PFI0), so FINITE mode completes
# after `samps_per_chan` edges regardless -- this rate just has to be <= the device max and
# match the real ~400 kHz FPGA clock. Verified live 2026-06-02 (first physical pyctrl run).

# Per-task metadata (rate + external clock source) kept OFF the nidaqmx Task: nidaqmx>=1.x made
# Task a __slots__ class with no __dict__, so attaching `task._yb_rate = ...` raises
# AttributeError ('Task' object has no attribute '_yb_rate'). A WeakKeyDictionary keyed by the
# task carries it instead -- auto-evicted when the task is closed/GC'd (Task is weakref-able:
# __weakref__ is in its __slots__). Version-robust across nidaqmx 1.0.x (no slots) and 1.5+.
_TASK_META = weakref.WeakKeyDictionary()

# Per-task ARMED sample count: the `samps_per_chan` the task's sample-clock timing is currently
# configured (and COMMITTED) for. `_write_and_start` re-runs cfg_samp_clk_timing ONLY when this
# does not match the incoming block -- see :func:`_write_and_start` for why that matters. Same
# WeakKeyDictionary pattern (and lifetime) as _TASK_META; a rebuilt task starts with no entry,
# so the first arm always configures. INVALIDATED (popped) on every error path that stops the
# task, so a task whose DAQmx state may have been reset by an error is always reconfigured.
_TASK_ARMED = weakref.WeakKeyDictionary()

# DAQmx -200018: "DAC conversion attempted before data to be converted was available".
# Same matching logic as lib/run_seq.py _is_transient_ni -- duplicated here deliberately so a
# devices/ module never imports lib/.
_NI_UNDERFLOW_CODE = -200018


def _is_underflow(exc):
    """True if ``exc`` is the DAQmx -200018 AO underflow (code OR message text).

    The ``error_code`` coercion is guarded: this runs INSIDE an ``except`` handler, so a weird
    (non-numeric, property-raising) ``error_code`` must not replace the original exception with
    a TypeError/ValueError -- it falls through to the message-text check instead.
    """
    try:
        if int(getattr(exc, "error_code", 0) or 0) == _NI_UNDERFLOW_CODE:
            return True
    except (ValueError, TypeError):
        pass
    msg = str(exc)
    return str(_NI_UNDERFLOW_CODE) in msg or "DAC conversion attempted before data" in msg


class NiDAQRunner:
    """Process-global NI session with structural-equality caching (NiDAQRunner.m)."""

    _session = None            # the nidaqmx Task (or a stub in tests)
    _cache_in_use = False
    _channels = None           # cached channel list (structural copy)
    _clocks = {}
    _triggers = {}
    _last_nsamps = None        # samples/chn armed for the in-flight shot (wait()'s triage)
    _gen_base = None           # generated-count snapshot taken right AFTER start (delta gate)
    underflows_swallowed = 0   # observability: cosmetic -200018s triaged away this session

    # ----------------------------------------------------------------------- #
    # public surface (mirrors NiDAQRunner.run / wait / clear_session)
    # ----------------------------------------------------------------------- #
    @classmethod
    def run(cls, channels, clocks, triggers, data):
        """Arm one shot: (re)acquire the session, STOP it (FINITE re-arm), write the
        channel-major data, and start it in the background (does NOT wait -- ``wait`` is
        called after the engine's own wait loop, in run_bseq)."""
        task = cls._get_session(channels, clocks, triggers)
        samples = _to_channel_major(data)        # [nsamps, nchns] -> [nchns, nsamps]
        # Armed sample count for wait()'s -200018 triage. Recorded HERE (not in the
        # NEEDS-HARDWARE _write_and_start hook, which tests stub out) so it is test-visible.
        cls._last_nsamps = _nsamps_of(samples)
        # Base for wait()'s DELTA gate. `total_samp_per_chan_generated` has UNDOCUMENTED reset
        # semantics across stop/write/start on a REUSED Task (pyctrl caches ONE Task for every
        # shot): it may be cumulative over the task's whole lifetime, or reset per start, or
        # reset by the error auto-stop. Snapshotting it right AFTER the start and gating on
        # (generated - base) is correct under ALL of those: cumulative -> the delta is this
        # shot's count; per-start reset -> base is ~0 and the delta equals generated. A raw
        # `generated >= nsamps` gate would be trivially true from shot 2 onward on a cumulative
        # counter (swallowing GENUINE underflows). Cleared first so a failed arm cannot leave a
        # stale base behind.
        cls._gen_base = None
        try:
            _write_and_start(task, samples)
            cls._gen_base = _generated_count(task)
        except BaseException as e:
            if _is_underflow(e):
                # A -200018 latched by the PREVIOUS shot's post-sequence PFI0 clock burst can
                # surface HERE, at the first DAQmx call of the new shot (the unguarded
                # task.stop() inside _write_and_start). The samples of that prior shot were
                # already converted; the error is only sticky DAQmx state on the task.
                _log.warning(
                    "[nidaq] stale DAQmx %d at re-arm from prior shot's clock burst; "
                    "rebuilding NI task and retrying once (nsamps=%s): %s",
                    _NI_UNDERFLOW_CODE, cls._last_nsamps, e)
                # cache_in_use is set -> _get_session closes the errored task and builds a
                # FRESH one (and refreshes the cached structural state) before the retry.
                task = cls._get_session(channels, clocks, triggers)
                try:
                    _write_and_start(task, samples)
                    cls._gen_base = _generated_count(task)   # re-armed -> re-snapshot the base
                    return
                except BaseException:
                    _stop_task(task)
                    raise
            # A started-but-unwaited FINITE task blocks the next shot: stop/reset + force a
            # rebuild next time (cache_in_use stays True -> get_session rebuilds).
            _stop_task(task)
            raise

    @classmethod
    def wait(cls):
        """Busy-wait for the in-flight shot to finish, then release the cache.

        ``cache_in_use`` is cleared ONLY here -- so an error between ``run`` and ``wait``
        leaves it set and the next ``run`` rebuilds the session (matches MATLAB).

        Triage of DAQmx -200018: the sequence emits EXACTLY N clock edges, then the molecube2
        ZYNQ server floods PFI0 with a ~10 ms / ~5 MHz burst (~50 k edges, 200 ns apart) after
        EVERY sequence (lib/controller.cpp:716-736, "a hack that is believed to make the NI card
        happy"). The first burst edge is the N+1th edge that COMPLETES the finite task; if the
        card loses the ~200 ns retire race, the next burst edge strobes an empty FIFO ->
        -200018 with all N real samples already converted.

        The gate is a DELTA against the base snapshotted right after this shot's start (run()),
        NOT the raw count: ``total_samp_per_chan_generated`` has undocumented reset semantics on
        the reused Task, so only ``generated - base >= nsamps`` is meaningful under BOTH
        cumulative and per-start-reset counters. It also requires ``preload_ok``: the "all real
        samples already converted" argument holds only when the whole waveform sat in the
        onboard FIFO -- on the best-effort streaming fallback a GENUINE mid-sequence host-feed
        underflow could still end with generated >= nsamps. None-handling is strict: an
        unreadable base or count falls through to the re-raise path (today's behavior).
        Anything else (short count, a different error) is a genuine fault and propagates."""
        task = cls._session
        if task is not None:
            try:
                _wait_task(task)
            except Exception as e:  # noqa: BLE001 - triage the known burst-race underflow
                if not _is_underflow(e):
                    raise
                nsamps = getattr(cls, "_last_nsamps", None)
                base = getattr(cls, "_gen_base", None)
                generated = _generated_count(task)
                preload_ok = _TASK_META.get(task, (None, None, True))[2]
                if (nsamps is not None and base is not None and generated is not None
                        and (generated - base) >= nsamps and preload_ok):
                    # Post-sequence clock-burst race (molecube2 controller.cpp:716-736): all
                    # real samples were already converted (the engine wait returned first; the
                    # onboard FIFO had every real edge covered), so the shot's analog output is
                    # complete and this error is cosmetic. Swallow it and keep the scan running.
                    # cache_in_use is CLEARED (task reused): leaving it True forced a ~0.7 s
                    # full channel rebuild at the NEXT bseq arm, which in a multi-bseq
                    # (rearrangement) shot lands MID-SHOT while atoms are held. Instead stop the
                    # task and reuse it; if it really is poisoned by sticky DAQmx error state,
                    # the next _write_and_start raises a stale -200018 and run()'s existing
                    # rebuild-and-retry-once path recovers -- so the rebuild cost is paid only
                    # when actually needed, never unconditionally mid-shot.
                    _stop_task(task)               # best-effort error-state clear
                    cls.underflows_swallowed += 1
                    _log.warning(
                        "[nidaq] DAQmx %d after the post-sequence PFI0 clock burst: "
                        "generated=%s-%s=%s/%s samples per chn (preload_ok=%s) -- shot output "
                        "COMPLETE, error swallowed (%d this session); NI task stopped + reused",
                        _NI_UNDERFLOW_CODE, generated, base,
                        (generated - base), nsamps, preload_ok, cls.underflows_swallowed)
                    cls._cache_in_use = False
                    return
                # Genuine (or unverifiable) underflow. Stop the errored task before re-raising:
                # an open+errored Task keeps its AO channels DAQmx-RESERVED, so out-of-band
                # dashboard DC sets would fail with -50103 until the idle release. cache_in_use
                # stays True -> the next _get_session rebuilds a FRESH task.
                _stop_task(task)
                _log.warning(
                    "[nidaq] DAQmx %d NOT attributable to the clock burst: generated=%s "
                    "base=%s nsamps=%s samples per chn (preload_ok=%s) -- real underflow or "
                    "count unqueryable, re-raising",
                    _NI_UNDERFLOW_CODE, generated, base, nsamps, preload_ok)
                raise
        cls._cache_in_use = False

    @classmethod
    def clear_session(cls):
        if cls._session is not None:
            _close_task(cls._session)
            cls._session = None

    @classmethod
    def has_session(cls):
        """True iff a (cached) NI Task is currently open.

        While a Task is open DAQmx keeps its AO channels RESERVED (even stopped/idle), so any
        out-of-band writer in another process -- e.g. the dashboard's one-off DC set via
        ``nidaq_io_handler.set_channel`` -- is refused with DAQmx -50103 "resource is reserved".
        The consume loop uses this to release the session when the backend goes truly idle."""
        return cls._session is not None

    # ----------------------------------------------------------------------- #
    # session cache (structural invalidation)
    # ----------------------------------------------------------------------- #
    @classmethod
    def _get_session(cls, channels, clocks, triggers):
        need_rebuild = (
            cls._cache_in_use
            or cls._session is None
            or not _channels_equal(channels, cls._channels)
            or not _map_equal(cls._clocks, clocks)
            or not _map_equal(cls._triggers, triggers))
        if need_rebuild:
            _close_task(cls._session)
            cls._session = _build_task(channels, clocks, triggers, _RATE)
            cls._channels = _copy_channels(channels)
            cls._clocks = dict(clocks)
            cls._triggers = dict(triggers)
        cls._cache_in_use = True
        return cls._session

    @classmethod
    def reset_cache(cls):
        """Drop all cached state WITHOUT touching hardware (tests / a clean re-init)."""
        cls._session = None
        cls._cache_in_use = False
        cls._channels = None
        cls._clocks = {}
        cls._triggers = {}
        cls._last_nsamps = None
        cls._gen_base = None
        cls.underflows_swallowed = 0
        _TASK_ARMED.clear()


# =========================================================================== #
# pure helpers (NO-HARDWARE)
# =========================================================================== #
def _to_channel_major(data):
    """``[nsamps, nchns]`` -> ``[nchns, nsamps]`` (the nidaqmx write order). #1 silent bug."""
    try:
        import numpy as np
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            raise ValueError("NI DAQ data must be 2-D [nsamps, nchns], got ndim=%d" % arr.ndim)
        # arr.T is an F-contiguous VIEW; nidaqmx's write requires a C-contiguous buffer.
        return np.ascontiguousarray(arr.T)
    except ImportError:
        # numpy-free fallback: data is a list of nsamps rows, each [ch0..chN].
        nsamps = len(data)
        nchns = len(data[0]) if nsamps else 0
        return [[float(data[s][c]) for s in range(nsamps)] for c in range(nchns)]


def _nsamps_of(samples):
    """Samples per channel of a channel-major ``[nchns, nsamps]`` block (array or list)."""
    if hasattr(samples, "shape"):
        return int(samples.shape[1])
    return len(samples[0]) if len(samples) else 0


def _chan_key(ch):
    """Structural identity of one AO channel -- (dev, chn). Accepts a dict or an object."""
    if isinstance(ch, dict):
        return (ch.get("dev"), ch.get("chn"))
    return (getattr(ch, "dev", None), getattr(ch, "chn", None))


def _channels_equal(a, b):
    """``isequaln`` of two channel lists by (dev, chn) in ORDER (add-order is load-bearing)."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    return all(_chan_key(x) == _chan_key(y) for x, y in zip(a, b))


def _map_equal(a, b):
    """``map_equal``: same keys and same values (dict equality)."""
    return dict(a) == dict(b)


def _copy_channels(channels):
    return list(channels)


# =========================================================================== #
# device hooks -- lazily import nidaqmx; only ever called on a real run (NEEDS-HARDWARE).
# Tests stub these so the cache decision + transpose are exercised NO-HARDWARE.
# =========================================================================== #
def _build_task(channels, clocks, triggers, rate):
    """Create + configure a nidaqmx AO Task to LISTEN on the external clock/trigger.

    Adds AO channels in ORDER (channel i of the data = AO channel i), configures a digital
    start trigger on PFI1, and stashes the rate + external sample-clock source (PFI0) for the
    per-shot timing config in :func:`_write_and_start`. A Task has ONE start trigger + ONE
    sample clock, so they are configured from the first channel's device -- correct for the
    single PCIe-6738 (Dev1); multi-device sync is out of scope. (NEEDS-HARDWARE.)
    """
    import nidaqmx                                   # noqa: F401 - lazy, hardware-only
    from nidaqmx.constants import Edge

    task = nidaqmx.Task()
    clk_src = None
    preload_ok = True
    for ch in channels:
        dev, chn = _chan_key(ch)
        ao = task.ao_channels.add_ao_voltage_chan("%s/ao%s" % (dev, chn))
        # Preload the WHOLE finite waveform into the card's 65,535-sample onboard FIFO, so there
        # is NO host->FIFO DMA during the externally-clocked (FPGA PFI0) generation. That removed
        # the HOST-FEED component of the intermittent DAQmx -200018 ("DAC conversion attempted
        # before data ... available"): the host can no longer be late with a sample. Our waveforms
        # are ~210 samples x 14 chn (~3 k) << the 65,535 shared FIFO, so the preload fits easily.
        # It is NOT the whole fix, though -- source-verified 2026-08-17: the sequence emits EXACTLY
        # N clock edges (zero margin) and the molecube2 ZYNQ server then deliberately floods PFI0
        # with a ~10 ms burst at ~5 MHz (~50 k edges, 200 ns apart) after EVERY sequence ("a hack
        # that is believed to make the NI card happy", molecube2 lib/controller.cpp:716-736). The
        # first burst edge is the N+1th edge that COMPLETES the finite task; the residual -200018 is
        # the card losing that ~200 ns retire race, so a following burst edge strobes the DAC with
        # an empty FIFO -- with all N real samples already converted. NiDAQRunner.wait() triages
        # that case (generated >= nsamps) and the scan continues. The MATLAB driver's Rate=500e3
        # "trailing-edge margin" never existed at the DAQmx layer either (see the clamp below).
        # Best-effort: a card/nidaqmx that rejects the property falls back to the streaming path
        # (the old behavior) -- recorded in _TASK_META so wait() can report it.
        try:
            ao.ao_use_only_on_brd_mem = True
        except Exception:  # noqa: BLE001 - unsupported -> keep the streaming path
            preload_ok = False
        if clk_src is None:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                "/%s/%s" % (dev, triggers[dev]), trigger_edge=Edge.RISING)
            clk_src = "/%s/%s" % (dev, clocks[dev])   # external sample clock (PFI0)
    # With an EXTERNAL clock, `rate` is only DAQmx's expected-max hint (buffer sizing + the
    # device's computed max-rate); the real timing is the FPGA PFI0 edges. This clamp was meant to
    # restore the MATLAB driver's Rate=500e3 "trailing-edge margin" from the device's OWN ceiling
    # instead of a hard-coded value the 6738 rejects (-200332). MEASURED on the PCIe-6738
    # 2026-08-17: samp_clk_max_rate is 1 MHz at ONE channel but 400 kHz for ANY >= 2 channels,
    # i.e. exactly _RATE -- so on this card `dev_max > rate` never fires and the clamp is dead
    # code (kept: harmless, and other cards may have real headroom). The MATLAB "margin" never
    # existed at the DAQmx layer either: cfg at 450k / 500k / 1M is rejected with -200332.
    try:
        dev_max = float(task.timing.samp_clk_max_rate)
        if dev_max > rate:
            rate = dev_max
    except Exception:  # noqa: BLE001 - query unsupported -> keep the nominal rate
        pass
    _TASK_META[task] = (rate, clk_src, preload_ok)
    return task


def _write_and_start(task, samples):
    """STOP (FINITE re-arm) -> set per-shot FINITE timing IF IT CHANGED -> write channel-major
    samples -> start (no wait). NEEDS-HARDWARE.

    ``samps_per_chan`` is the data length, which varies per bseq -- so the FINITE sample-clock
    timing is (re)configured here, mirroring MATLAB ``queueOutputData`` inferring the count from
    the queued data (the cached Task keeps the expensive channel/trigger setup).

    **Reconfigured only on a CHANGE, then COMMITTED** (2026-09-17): writing any timing property
    knocks a DAQmx task out of the COMMITTED state, so the following ``start()`` has to redo
    verify -> reserve -> commit (route the PFI lines, size the buffer, program the DMA/FIFO)
    from scratch on EVERY arm. ``stop -> write -> start`` on a task that STAYS committed is the
    documented fast re-arm path, and ``DAQmxStopTask`` returns a task to the state it held
    before ``start()`` -- i.e. COMMITTED -- so the commit survives the per-shot stop. Guarding
    the cfg call on ``_TASK_ARMED`` and committing once buys that path whenever a bseq's sample
    count repeats (the common case: only FPGA/AWG axes are usually swept, not the NI block
    length). A changed count simply pays the old cost once. Measured motivation: ni_arm was
    ~28 ms x 3 bseqs = 83 ms/shot on job 201 (RearrangeSTIRAPScan, 2026-09-17).

    The commit is best-effort: a card or nidaqmx build that refuses TASK_COMMIT falls back to
    the previous behavior (implicit commit inside ``start()``), only slower -- never wrong.
    """
    from nidaqmx.constants import AcquisitionType, Edge
    nsamps = _nsamps_of(samples)
    task.stop()
    if _TASK_ARMED.get(task) != nsamps:
        meta = _TASK_META[task]
        rate, clk_src = meta[0], meta[1]         # meta = (rate, clk_src, preload_ok)
        task.timing.cfg_samp_clk_timing(
            rate, source=clk_src, active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE, samps_per_chan=nsamps)
        # Recorded only AFTER cfg returned: if cfg raises, the entry keeps its old (different)
        # value, so the next arm reconfigures -- which is what we want on an errored task.
        _commit_task(task)
        _TASK_ARMED[task] = nsamps
    # nidaqmx wants a 1-D array for a SINGLE-channel task; a (1, nsamps) 2-D array is misread
    # (DaqError -200524, "number of channels in the data does not match"). Squeeze the lone
    # channel axis -- harmless for the multi-channel path (left untouched). Seen first on the
    # 1-NI-channel PicoMotor308 seq (2026-06-26).
    write_data = samples
    if hasattr(samples, "shape"):
        if samples.shape[0] == 1:
            write_data = samples[0]
    elif len(samples) == 1:
        write_data = samples[0]
    task.write(write_data, auto_start=False)
    task.start()


def _wait_task(task):
    task.wait_until_done()


def _generated_count(task):
    """Total samples per channel actually converted by the card (None if unqueryable)."""
    try:
        return int(task.out_stream.total_samp_per_chan_generated)
    except Exception:  # noqa: BLE001 - property may be unreadable on an errored/closed task
        return None


def _commit_task(task):
    """Force the task to the COMMITTED state now (best-effort). NEEDS-HARDWARE.

    ``DAQmxTaskControl(TASK_COMMIT)`` does the verify/reserve/program-hardware work up front so
    the per-shot ``start()`` is a bare arm. Swallowed on failure -- an older nidaqmx without
    ``TaskMode``, or a device that refuses the transition, just keeps the implicit-commit-in-
    ``start()`` behavior: slower, never incorrect.
    """
    try:
        from nidaqmx.constants import TaskMode
        task.control(TaskMode.TASK_COMMIT)
    except Exception:  # noqa: BLE001 - optional fast path only
        pass


def _stop_task(task):
    """Best-effort stop + FORGET the armed sample count.

    The forget is load-bearing: every caller of this is an error path, and a DAQmx error can
    reset the task's timing/committed state under us. Dropping the cache entry makes the next
    arm reconfigure + recommit unconditionally rather than trusting state we cannot verify.
    """
    _TASK_ARMED.pop(task, None)
    try:
        task.stop()
    except Exception:  # noqa: BLE001 - best-effort on the error path
        pass


def _close_task(task):
    if task is None:
        return
    _TASK_ARMED.pop(task, None)      # the WeakKeyDictionary would evict on GC anyway; explicit
    try:                             # so a still-referenced closed task can never be trusted
        task.close()
    except Exception:  # noqa: BLE001
        pass
