"""awg_manager.py -- port of ``matlab_new/YbExptCtrl/sigilentAWG/AWGManager.m``.

The scan-long Siglent AWG coordinator. Two phases, mirroring MATLAB:

  * :meth:`AWGManager.setup` (BATCH UPLOAD, once at scan start / dequeue) -- for each named AWG,
    loads its ``Consts()`` defaults, walks every sequence in the ScanGroup to find the UNIQUE
    waveform-shaping combos, connects, and (per the box's firmware capability) either PRE-STORES
    every unique waveform under a stable name (``ARWV`` path, fw >= 6.01.01.38) or pre-builds one
    ``WVDT`` command per combo (re-upload fallback, older fw). Sets amplitude once + arms gated burst.
  * :meth:`AWGManager.recall_for_seq` (PER-SHOT SWITCH, once per sequence) -- switches to this
    shot's waveform, skipping when the waveform key is unchanged. **ARWV path:** ``ARWV NAME,<name>``
    (~ms, no re-upload; the stored waveform carries its baked ``FREQ`` so width comes along).
    **Fallback:** re-send the full pre-built ``WVDT`` to the ``active`` slot (~2 ms). The path is
    chosen at ``setup`` from ``connection.arwv_recall`` (problem-memory ``bug-awg-arwv-broken``:
    37R6 ARWV broken -> 38R3 fixed; AWG308 may still be on 37R6 -> fallback).
  * :meth:`AWGManager.cleanup` -- disconnect all and clear state.

State is **process-global** (class attribute ``_state``), the faithful analog of the MATLAB
``persistent`` var: the per-shot ``reg_before_start`` callback only receives ``s1``, so it reaches
the live connections + command cache through these classmethods (just like ``rearrange_runtime``
backs the SLM per-shot callbacks). One scan at a time owns the AWGs.

Waveform-shaping fields (different value -> different uploaded waveform):
``carrier_freq_MHz``, ``pulse_width_us``, ``steepness``, ``amplitude_scale``.
Hardware-config fields (read once, never change the waveform data):
``resource_address``, ``channel``, ``max_amplitude_vpp``, ``num_points``.

``setup`` takes injectable ``consts`` + ``connection_factory`` seams so the batch-upload / dedup
logic is unit-testable with a fake connection and NO hardware (see ``test_sigilent_awg.py``).
"""
import logging
import time

from .awg_connection import AWGConnection
from .gaussian_pulse_waveform import gaussian_pulse_waveform

logger = logging.getLogger(__name__)

WAVEFORM_FIELDS = ("carrier_freq_MHz", "pulse_width_us", "steepness", "amplitude_scale")


class AWGManager:
    # Process-global scan state: awg_name -> entry dict
    #   {connection, cmd_map, awg_name, defaults, last_key}
    _state = {}

    # --------------------------------------------------------------------- #
    # phase 1: batch upload (once per scan)
    # --------------------------------------------------------------------- #
    @classmethod
    def setup(cls, awg_names, scangroup, *, consts=None, connection_factory=None):
        """Pre-build + upload every unique waveform for ``awg_names`` across the whole scan.

        Args:
            awg_names: an AWG name (``"AWG556"``) or an iterable of names.
            scangroup: a :class:`ScanGroup` exposing ``nseq()`` + ``getseq(n)`` (1-based).
            consts: optional ``{awg_name: defaults_dict}`` mapping. Defaults to the live config
                (``SeqConfig.get().consts``).
            connection_factory: optional ``(resource, channel) -> connection`` (the connection must
                provide ``connect/build_waveform_cmd/send_waveform/set_amplitude/configure_burst/
                enable_output/disconnect``). Defaults to :class:`AWGConnection` (real USB-VISA).
        """
        if isinstance(awg_names, str):
            awg_names = [awg_names]
        consts_src = consts if consts is not None else cls._live_consts()
        make_conn = connection_factory or AWGConnection

        state = {}
        total_seqs = scangroup.nseq()
        for awg_name in awg_names:
            defaults = dict(consts_src[awg_name])

            # Collect unique waveform-shaping combos across all sequences.
            keys = []
            param_list = []
            for n in range(1, total_seqs + 1):
                seq = scangroup.getseq(n)
                params = dict(defaults)
                params.update(cls._seq_awg_overrides(seq, awg_name))
                key = cls._build_key(params)
                if key not in keys:
                    keys.append(key)
                    param_list.append(params)

            conn = make_conn(defaults["resource_address"], defaults["channel"])
            conn.connect()
            use_arwv = bool(getattr(conn, "arwv_recall", False))

            amp_vpp = param_list[0]["max_amplitude_vpp"]
            num_points = defaults["num_points"]

            logger.info("AWGManager: %s -- %d unique waveform(s) for %d sequences (mode=%s)",
                        awg_name, len(keys), total_seqs, "arwv" if use_arwv else "reupload")

            entry = {
                "connection": conn,
                "awg_name": awg_name,
                "defaults": defaults,
                "mode": "arwv" if use_arwv else "reupload",
                "last_key": keys[0] if keys else None,
            }

            if use_arwv:
                # ARWV path (fw >= 38R3): pre-store every unique waveform under a stable name,
                # switch per shot with ARWV NAME (no re-upload). ARWV restores the baked FREQ.
                conn.set_arb_mode()
                name_map = {}
                for i, key in enumerate(keys):
                    p = dict(param_list[i])
                    p["max_amplitude_vpp"] = amp_vpp
                    p["num_points"] = num_points
                    binary_data, info = gaussian_pulse_waveform(p)
                    name = "wf_%03d" % i
                    conn.store_waveform(
                        conn.build_waveform_cmd(binary_data, amp_vpp, info["freq_hz"], name=name))
                    name_map[key] = name
                    logger.info("  %s <- %d pts, freq=%gHz, key: %s",
                                name, info["num_points"], info["freq_hz"], key)
                conn.set_amplitude(amp_vpp)
                conn.configure_burst()
                conn.enable_output()
                if keys:
                    conn.recall_by_name(name_map[keys[0]])
                entry["name_map"] = name_map
            else:
                # Re-upload fallback (older fw, e.g. AWG308 on 37R6): cache one WVDT per combo,
                # re-send the active-slot WVDT per shot.
                cmd_map = {}
                for i, key in enumerate(keys):
                    p = dict(param_list[i])
                    p["max_amplitude_vpp"] = amp_vpp
                    p["num_points"] = num_points
                    binary_data, info = gaussian_pulse_waveform(p)
                    cmd_map[key] = conn.build_waveform_cmd(binary_data, amp_vpp, info["freq_hz"])
                    logger.info("  wf_%03d: %d pts, freq=%gHz, key: %s",
                                i + 1, info["num_points"], info["freq_hz"], key)
                if keys:
                    conn.send_waveform(cmd_map[keys[0]])   # init output
                    time.sleep(0.05)
                conn.set_amplitude(amp_vpp)
                conn.configure_burst()
                conn.enable_output()
                entry["cmd_map"] = cmd_map

            state[awg_name] = entry

        cls._state = state
        logger.info("AWGManager: setup complete (%s)", ", ".join(awg_names))

    # --------------------------------------------------------------------- #
    # phase 2: per-shot active-waveform switch
    # --------------------------------------------------------------------- #
    @classmethod
    def recall_for_seq(cls, awg_struct):
        """Switch each active AWG to this shot's waveform (~2 ms), or no-op if unchanged.

        Args:
            awg_struct: a mapping ``{awg_name: {field: value, ...}, ...}`` of this shot's AWG
                params (the per-shot scan values, e.g. read from ``s1.C.AWG``). Missing AWGs /
                fields fall back to the setup-time defaults. No-op when no scan configured AWGs.
        """
        if not cls._state:
            return
        awg_struct = awg_struct or {}
        for awg_name, entry in cls._state.items():
            params = dict(entry["defaults"])
            overrides = awg_struct.get(awg_name)
            if overrides:
                params.update(overrides)

            key = cls._build_key(params)
            if key == entry["last_key"]:
                continue                      # unchanged from last shot -> skip the switch
            if entry["mode"] == "arwv":
                name = entry["name_map"].get(key)
                if name is not None:
                    entry["connection"].recall_by_name(name)   # ARWV NAME -- ~ms, no re-upload
                    entry["last_key"] = key
                else:
                    logger.warning("AWGManager: no stored waveform for %s key: %s", awg_name, key)
            else:
                cmd = entry["cmd_map"].get(key)
                if cmd is not None:
                    entry["connection"].send_waveform(cmd)     # fallback: re-send WVDT to active
                    entry["last_key"] = key
                else:
                    logger.warning("AWGManager: no waveform for %s key: %s", awg_name, key)

    # --------------------------------------------------------------------- #
    # teardown
    # --------------------------------------------------------------------- #
    @classmethod
    def cleanup(cls):
        for awg_name, entry in cls._state.items():
            conn = entry["connection"]
            try:
                conn.disable_output()       # OUTP OFF + burst OFF -> AWG quiet after the scan
            except Exception as err:        # noqa: BLE001
                logger.warning("AWGManager: error disabling output %s: %s", awg_name, err)
            try:
                conn.disconnect()
                logger.info("AWGManager: disconnected %s", awg_name)
            except Exception as err:        # noqa: BLE001
                logger.warning("AWGManager: error disconnecting %s: %s", awg_name, err)
        cls._state = {}

    @classmethod
    def active_awgs(cls):
        """Names of the AWGs currently set up (for diagnostics / the run loop)."""
        return sorted(cls._state)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _build_key(params):
        """String key from the waveform-shaping fields present (matches AWGManager.m::buildKey)."""
        parts = []
        for field in WAVEFORM_FIELDS:
            if field in params and params[field] is not None:
                parts.append("%s=%.8g" % (field, float(params[field])))
        return "|".join(parts)

    @staticmethod
    def _seq_awg_overrides(seq, awg_name):
        """Extract ``seq.AWG.<awg_name>`` overrides from a getseq() result (empty dict if none)."""
        if not isinstance(seq, dict):
            return {}
        awg = seq.get("AWG")
        if isinstance(awg, dict) and isinstance(awg.get(awg_name), dict):
            return dict(awg[awg_name])
        return {}

    @staticmethod
    def _live_consts():
        """Default consts source: the live SeqConfig consts tree (``{awg_name: dict}``)."""
        from seq_config import SeqConfig
        return SeqConfig.get().consts
