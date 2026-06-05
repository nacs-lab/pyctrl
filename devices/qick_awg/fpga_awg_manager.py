"""fpga_awg_manager.py -- QICK FPGA_AWG scan coordinator: BATCH UPLOAD + PULSE PICKING.

The Siglent ``AWGManager`` analog for the QICK board. The QICK socket protocol already supports the
two pieces (uploads are named + persistent; ``start_program(name)`` selects which resident program
plays) -- but the existing MATLAB usage (``RamseySeq``/``PushoutSurvivalAWGSeq`` ``server_pre_run``)
does NEITHER: it ``delete_all_*`` + re-uploads pulses+program and ``start_program``\\ s **every shot**.
This manager adds the optimization, mirroring the Siglent pattern:

  * :meth:`setup` (BATCH UPLOAD, once per scan) -- collect the UNIQUE programs across the scan (by
    ``key``), connect, ``delete_all_*``, then upload each unique program once. Pulses are namespaced
    per program (``p000_Pi2`` ...) so multiple programs coexist on the server without name clashes;
    the prog_structure tokens are rewritten to match via :func:`compile_chn`'s ``name_map``. Sets the
    trigger mode and arms the first program.
  * :meth:`recall_for_seq` (PER-SHOT PICK, once per sequence) -- ``stop_program`` + ``start_program``
    for this shot's program, or a no-op when the key is unchanged from the previous shot.
  * :meth:`cleanup` -- ``stop_program`` + disconnect.

State is **process-global** (class attribute ``_state``), the faithful analog of the MATLAB
``global fpga`` -- the per-shot callback only receives the seq, so it reaches the live connection +
program map through these classmethods (like the Siglent ``AWGManager`` and ``rearrange_runtime``).

⚠ Out-of-band device: QICK output is NOT in the serialized byte blob, so THE ONE RULE does not apply.
The per-seq program selection (what ``recall_for_seq`` is handed) and the run-loop wiring are a
Phase-6 scan-convention decision and are deliberately NOT wired into ``runner.py`` here.
"""
import json
import logging

from .fpga_awg_client import DEFAULT_HOST, DEFAULT_PORT, FPGAAWGClient
from .simple_pulse import compile_chn, simple_prog_cfg

logger = logging.getLogger(__name__)


class QickProgram:
    """One uploadable QICK program: a selection ``key``, its ``pulses``, and its ``channels``.

    Args:
        key: the dedup/selection key (any hashable) -- programs sharing a key are uploaded once and
            picked together (e.g. the swept microwave freq/gain/phase tuple).
        pulses: ``{pulse_name: pulse_cfg_dict}`` (build cfgs with :func:`simple_pulse_cfg`).
        channels: list of 1 or 2 token lists referencing those pulse names (see :mod:`simple_pulse`).
    """

    __slots__ = ("key", "pulses", "channels")

    def __init__(self, key, pulses, channels):
        self.key = key
        self.pulses = dict(pulses)
        self.channels = list(channels)


def _json_bytes(obj):
    return json.dumps(obj).encode("utf-8")


class FPGAAWGManager:
    # Process-global scan state:
    #   {client, key_to_progname: {key: prog_name}, last_key, host, port}
    _state = {}

    # --------------------------------------------------------------------- #
    # phase 1: batch upload (once per scan)
    # --------------------------------------------------------------------- #
    @classmethod
    def setup(cls, programs, *, host=DEFAULT_HOST, port=DEFAULT_PORT,
              client_factory=None, trigger_mode="external"):
        """Upload every UNIQUE program (by ``key``) once and arm the first.

        Args:
            programs: an iterable of :class:`QickProgram` (one per scan point, in order; duplicates
                by ``key`` are uploaded once).
            host/port: the QICK server address (defaults to the lab server).
            client_factory: ``() -> client`` providing the ``FPGAAWGClient`` surface (connect /
                delete_all_* / upload_* / set_trigger_mode / start_program / stop_program /
                disconnect). Defaults to :class:`FPGAAWGClient` (real socket). Injectable for tests.
            trigger_mode: passed to ``set_trigger_mode`` (``"external"`` -- FPGA1/TTL14 triggered).
        """
        programs = list(programs)
        make_client = client_factory or FPGAAWGClient

        # Unique programs, preserving first-seen order.
        unique = []
        seen = set()
        for prog in programs:
            if prog.key not in seen:
                seen.add(prog.key)
                unique.append(prog)

        client = make_client()
        client.connect(host, port)
        client.delete_all_envelope_data()
        client.delete_all_waveform_cfg()
        client.delete_all_programs()

        key_to_progname = {}
        for i, prog in enumerate(unique):
            prog_name = "prog%03d" % i
            # Namespace this program's pulses so they don't clash with other programs' pulses.
            name_map = {}
            for pname, cfg in prog.pulses.items():
                ns_name = "p%03d_%s" % (i, pname)
                name_map[pname] = ns_name
                ns_cfg = dict(cfg)
                ns_cfg["name"] = ns_name
                client.upload_waveform_cfg(_json_bytes(ns_cfg), ns_name)
            prog_structure = compile_chn(prog.channels, name_map=name_map)
            client.upload_program(_json_bytes(simple_prog_cfg(prog_name, prog_structure)), prog_name)
            key_to_progname[prog.key] = prog_name
            logger.info("  %s <- key %r (%d pulse(s))", prog_name, prog.key, len(prog.pulses))

        client.set_trigger_mode(trigger_mode)

        first_key = unique[0].key if unique else None
        if first_key is not None:
            client.start_program(key_to_progname[first_key])

        cls._state = {
            "client": client,
            "key_to_progname": key_to_progname,
            "last_key": first_key,
            "host": host,
            "port": port,
        }
        logger.info("FPGAAWGManager: setup complete -- %d unique program(s) for %d point(s)",
                    len(unique), len(programs))

    # --------------------------------------------------------------------- #
    # phase 2: per-shot program pick
    # --------------------------------------------------------------------- #
    @classmethod
    def recall_for_seq(cls, key):
        """Switch the active QICK program to ``key``'s, or no-op when unchanged from last shot."""
        if not cls._state:
            return
        if key == cls._state["last_key"]:
            return                                  # unchanged -> already armed, skip
        prog_name = cls._state["key_to_progname"].get(key)
        if prog_name is None:
            logger.warning("FPGAAWGManager: no uploaded program for key %r", key)
            return
        client = cls._state["client"]
        client.stop_program()
        client.start_program(prog_name)
        cls._state["last_key"] = key

    # --------------------------------------------------------------------- #
    # teardown
    # --------------------------------------------------------------------- #
    @classmethod
    def cleanup(cls):
        state = cls._state
        if state:
            client = state["client"]
            try:
                client.stop_program()
            except Exception as err:                # noqa: BLE001
                logger.warning("FPGAAWGManager: error stopping program: %s", err)
            try:
                client.disconnect()
            except Exception as err:                # noqa: BLE001
                logger.warning("FPGAAWGManager: error disconnecting: %s", err)
        cls._state = {}

    @classmethod
    def is_active(cls):
        """Whether a scan currently owns the QICK board (for diagnostics / the run loop)."""
        return bool(cls._state)
