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
``shape``, ``carrier_freq_MHz``, ``pulse_width_us``, ``smooth_width_us``, ``steepness``,
``amplitude_scale``, plus ``stirap_gap`` / ``f_delay`` / ``r_delay`` for the two-lobe STIRAP shapes
(``shape``/``smooth_width_us`` select the envelope family -- see :mod:`pulse_waveform`; extends
MATLAB AWGManager.m, which only had the symmetric Gaussian). Because these fields are in the key,
each scanned (gap, f_delay, r_delay) is pre-stored + recalled as its own named waveform.
Hardware-config fields (read once, never change the waveform data):
``resource_address``, ``channel``, ``max_amplitude_vpp``, ``num_points``.

``setup`` takes injectable ``consts`` + ``connection_factory`` seams so the batch-upload / dedup
logic is unit-testable with a fake connection and NO hardware (see ``test_sigilent_awg.py``).
"""
import logging
import time

from .awg_connection import AWGConnection
from .pulse_waveform import pulse_waveform

logger = logging.getLogger(__name__)

WAVEFORM_FIELDS = ("shape", "carrier_freq_MHz", "pulse_width_us", "smooth_width_us",
                   "steepness", "amplitude_scale",
                   "stirap_gap", "f_delay", "r_delay",   # two-lobe STIRAP shapes
                   "pad_time_us",                        # fall_quintic flat pre-hold
                   "chirp_freq_MHz", "chirp_profile")    # chirped_* swept carrier

# Per-channel sub-config keys (two-channel switch scheme). When a box's consts carry ``Ch1`` /
# ``Ch2`` dicts, each is an INDEPENDENT waveform on that SDG output (``C1`` / ``C2``), armed as a
# single-cycle externally-triggered burst; an external RF switch selects which reaches the AOM.
# Box-level fields shared by both channels (not per-channel waveform data):
CHANNEL_KEYS = ("Ch1", "Ch2")
BOX_SHARED_FIELDS = ("resource_address", "num_points", "sample_rate_MHz")


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

            # Two-channel switch scheme: box carries Ch1/Ch2 sub-dicts -> independent per-channel
            # waveforms on C1/C2, single-cycle EXT burst. Legacy flat config falls through below.
            if cls._is_channel_mode(defaults):
                state[awg_name] = cls._setup_box_channel(
                    awg_name, defaults, scangroup, total_seqs, make_conn)
                continue

            # ---- legacy single-channel path (unchanged) --------------------------------------
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
                    binary_data, info = pulse_waveform(p)
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
                    binary_data, info = pulse_waveform(p)
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
    # two-channel switch scheme: one box, independent C1/C2 waveforms
    # --------------------------------------------------------------------- #
    @classmethod
    def _setup_box_channel(cls, awg_name, defaults, scangroup, total_seqs, make_conn):
        """Set up a box in per-channel mode: build/upload a waveform per Ch1/Ch2, arm each as a
        single-cycle EXT burst. Returns the state entry ``{connection, mode:"channel", channels}``.

        ``Ch1``/``Ch2`` are waveform-shaping dicts (``shape``, ``carrier_freq_MHz``,
        ``pulse_width_us``, ..., plus ``channel`` = ``"C1"``/``"C2"``, ``max_amplitude_vpp``, and
        optional ``trig_delay_us`` = per-channel burst DLAY). Box-level ``num_points`` /
        ``sample_rate_MHz`` are shared into every channel's params."""
        box_shared = {k: defaults[k] for k in BOX_SHARED_FIELDS if k in defaults}
        ch_keys = [k for k in CHANNEL_KEYS if isinstance(defaults.get(k), dict)]

        conn = make_conn(defaults["resource_address"], defaults[ch_keys[0]].get("channel", "C1"))
        conn.connect()
        use_arwv = bool(getattr(conn, "arwv_recall", False))
        logger.info("AWGManager: %s -- channel mode (%s), %s",
                    awg_name, ", ".join(ch_keys), "arwv" if use_arwv else "reupload")

        entry = {"connection": conn, "awg_name": awg_name,
                 "mode": "channel", "channels": {}}

        for ck in ch_keys:
            chd = dict(box_shared)
            chd.update(defaults[ck])                       # per-channel waveform fields win
            scpi_ch = chd.get("channel", "C1" if ck == "Ch1" else "C2")
            dlay_s = float(chd.get("trig_delay_us", 0.0)) * 1e-6

            # Unique waveforms for THIS channel across the scan (per-channel scan overrides).
            keys, param_list = [], []
            for n in range(1, total_seqs + 1):
                params = dict(chd)
                params.update(cls._seq_awg_ch_overrides(scangroup.getseq(n), awg_name, ck))
                key = cls._build_key(params)
                if key not in keys:
                    keys.append(key)
                    param_list.append(params)

            # Amplitude is set ONCE per channel (BSWV AMP) and every stored waveform's AMPL must
            # match it, so it is not a WAVEFORM_FIELD and is uniform across param_list -> take it
            # from the MERGED per-seq params (param_list[0]), NOT raw chd, so a scan/scalar override
            # of max_amplitude_vpp is honored (legacy path does the same via param_list[0]).
            amp_vpp = param_list[0]["max_amplitude_vpp"] if param_list else chd["max_amplitude_vpp"]

            cmap = {}
            if use_arwv:
                conn.set_arb_mode(channel=scpi_ch)
                for i, key in enumerate(keys):
                    p = dict(param_list[i])
                    binary_data, info = pulse_waveform(p)
                    name = "%s_%s_%03d" % (awg_name, ck, i)
                    conn.store_waveform(conn.build_waveform_cmd(
                        binary_data, amp_vpp, info["freq_hz"], name=name, channel=scpi_ch))
                    cmap[key] = name
                    logger.info("  %s[%s] %s <- %d pts, freq=%gHz, key: %s",
                                awg_name, scpi_ch, name, info["num_points"], info["freq_hz"], key)
            else:
                for i, key in enumerate(keys):
                    p = dict(param_list[i])
                    binary_data, info = pulse_waveform(p)
                    cmap[key] = conn.build_waveform_cmd(
                        binary_data, amp_vpp, info["freq_hz"], channel=scpi_ch)
                    logger.info("  %s[%s] wf_%03d: %d pts, freq=%gHz, key: %s",
                                awg_name, scpi_ch, i, info["num_points"], info["freq_hz"], key)
                if keys:
                    conn.send_waveform(cmap[keys[0]])
                    time.sleep(0.05)

            conn.set_amplitude(amp_vpp, channel=scpi_ch)
            conn.configure_burst(channel=scpi_ch, mode="NCYC", ncyc=1, dlay=dlay_s)
            conn.enable_output(channel=scpi_ch)
            if use_arwv and keys:
                conn.recall_by_name(cmap[keys[0]], channel=scpi_ch)

            entry["channels"][ck] = {
                "scpi_ch": scpi_ch, "defaults": chd, "last_key": keys[0] if keys else None,
                ("name_map" if use_arwv else "cmd_map"): cmap, "use_arwv": use_arwv,
            }
        return entry

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
            if entry.get("mode") == "channel":
                cls._recall_channel(awg_name, entry, awg_struct.get(awg_name) or {})
                continue

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

    @classmethod
    def _recall_channel(cls, awg_name, entry, box_overrides):
        """Per-shot switch for a channel-mode box: each Ch1/Ch2 independently recalls this shot's
        waveform (skip if unchanged). ``box_overrides`` = ``{"Ch1": {...}, "Ch2": {...}}``."""
        conn = entry["connection"]
        for ck, chstate in entry["channels"].items():
            params = dict(chstate["defaults"])
            ov = box_overrides.get(ck) if isinstance(box_overrides, dict) else None
            if ov:
                params.update(ov)
            key = cls._build_key(params)
            if key == chstate["last_key"]:
                continue
            scpi_ch = chstate["scpi_ch"]
            if chstate["use_arwv"]:
                name = chstate["name_map"].get(key)
                if name is not None:
                    conn.recall_by_name(name, channel=scpi_ch)
                    chstate["last_key"] = key
                else:
                    logger.warning("AWGManager: no stored waveform for %s[%s] key: %s",
                                   awg_name, ck, key)
            else:
                cmd = chstate["cmd_map"].get(key)
                if cmd is not None:
                    conn.send_waveform(cmd)
                    chstate["last_key"] = key
                else:
                    logger.warning("AWGManager: no waveform for %s[%s] key: %s", awg_name, ck, key)

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
        """String key from the waveform-shaping fields present (extends AWGManager.m::buildKey
        with the string ``shape`` field + ``smooth_width_us``)."""
        parts = []
        for field in WAVEFORM_FIELDS:
            if field in params and params[field] is not None:
                v = params[field]
                if isinstance(v, str):
                    parts.append("%s=%s" % (field, v))
                else:
                    parts.append("%s=%.8g" % (field, float(v)))
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
    def _is_channel_mode(defaults):
        """True iff the box config carries per-channel ``Ch1``/``Ch2`` sub-dicts."""
        return any(isinstance(defaults.get(k), dict) for k in CHANNEL_KEYS)

    @staticmethod
    def _seq_awg_ch_overrides(seq, awg_name, ch_key):
        """Extract ``seq.AWG.<awg_name>.<ch_key>`` per-channel overrides (empty dict if none)."""
        if not isinstance(seq, dict):
            return {}
        awg = seq.get("AWG")
        if not (isinstance(awg, dict) and isinstance(awg.get(awg_name), dict)):
            return {}
        ch = awg[awg_name].get(ch_key)
        return dict(ch) if isinstance(ch, dict) else {}

    @staticmethod
    def _live_consts():
        """Default consts source: the live SeqConfig consts tree (``{awg_name: dict}``)."""
        from seq_config import SeqConfig
        return SeqConfig.get().consts
