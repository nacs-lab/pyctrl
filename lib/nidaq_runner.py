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

_RATE = 400e3          # NI sample-clock rate (Hz) = the FPGA PFI0 clock.
# ⚠ Was 500e3 ("over-estimate"), but the PCIe-6738 REJECTS >400 kHz with 14 channels
# (DaqError -200332, "Specified sample rate is higher than the fastest rate supported";
# device max = 400 kHz at 14 chn). The clock is EXTERNAL (PFI0), so FINITE mode completes
# after `samps_per_chan` edges regardless -- this rate just has to be <= the device max and
# match the real ~400 kHz FPGA clock. Verified live 2026-06-02 (first physical pyctrl run).


class NiDAQRunner:
    """Process-global NI session with structural-equality caching (NiDAQRunner.m)."""

    _session = None            # the nidaqmx Task (or a stub in tests)
    _cache_in_use = False
    _channels = None           # cached channel list (structural copy)
    _clocks = {}
    _triggers = {}

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
        try:
            _write_and_start(task, samples)
        except BaseException:
            # A started-but-unwaited FINITE task blocks the next shot: stop/reset + force a
            # rebuild next time (cache_in_use stays True -> get_session rebuilds).
            _stop_task(task)
            raise

    @classmethod
    def wait(cls):
        """Busy-wait for the in-flight shot to finish, then release the cache.

        ``cache_in_use`` is cleared ONLY here -- so an error between ``run`` and ``wait``
        leaves it set and the next ``run`` rebuilds the session (matches MATLAB)."""
        if cls._session is not None:
            _wait_task(cls._session)
        cls._cache_in_use = False

    @classmethod
    def clear_session(cls):
        if cls._session is not None:
            _close_task(cls._session)
            cls._session = None

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
    for ch in channels:
        dev, chn = _chan_key(ch)
        task.ao_channels.add_ao_voltage_chan("%s/ao%s" % (dev, chn))
        if clk_src is None:
            task.triggers.start_trigger.cfg_dig_edge_start_trig(
                "/%s/%s" % (dev, triggers[dev]), trigger_edge=Edge.RISING)
            clk_src = "/%s/%s" % (dev, clocks[dev])   # external sample clock (PFI0)
    task._yb_rate = rate
    task._yb_clk_src = clk_src
    return task


def _write_and_start(task, samples):
    """STOP (FINITE re-arm) -> set per-shot FINITE timing -> write channel-major samples ->
    start (no wait). NEEDS-HARDWARE.

    ``samps_per_chan`` is the data length, which varies per bseq -- so the FINITE sample-clock
    timing is (re)configured each shot here, mirroring MATLAB ``queueOutputData`` inferring the
    count from the queued data (the cached Task keeps the expensive channel/trigger setup).
    """
    from nidaqmx.constants import AcquisitionType, Edge
    nsamps = int(samples.shape[1]) if hasattr(samples, "shape") else len(samples[0])
    task.stop()
    task.timing.cfg_samp_clk_timing(
        task._yb_rate, source=task._yb_clk_src, active_edge=Edge.RISING,
        sample_mode=AcquisitionType.FINITE, samps_per_chan=nsamps)
    task.write(samples, auto_start=False)
    task.start()


def _wait_task(task):
    task.wait_until_done()


def _stop_task(task):
    try:
        task.stop()
    except Exception:  # noqa: BLE001 - best-effort on the error path
        pass


def _close_task(task):
    if task is None:
        return
    try:
        task.close()
    except Exception:  # noqa: BLE001
        pass
