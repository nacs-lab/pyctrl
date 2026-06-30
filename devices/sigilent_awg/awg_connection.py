"""awg_connection.py -- port of ``matlab_new/YbExptCtrl/sigilentAWG/AWGConnection.m``.

A thin USB-VISA wrapper around one Siglent SDG6X channel, in **DDS mode** (the WVDT ``FREQ``
controls playback rate). MATLAB used ``visadev``; here we use **pyvisa** over the same USB
resource string (e.g. ``USB0::62700::4353::SDG6XFCC900309::0::INSTR``). pyvisa is imported lazily
in :meth:`connect` so this module is import-safe with no VISA backend present (NO-HARDWARE).

SDG6X hard-won rules baked in (see the experiment-running skill / AWG_Integration_Plan.md):
  * **Big-endian int16** waveform bytes (produced by :mod:`gaussian_pulse_waveform`).
  * **ARWV recall was broken on firmware 6.01.01.37R6 -> FIXED on 6.01.01.38R3** (problem-memory
    ``bug-awg-arwv-broken``). When the connected box reports fw >= 6.01.01.38 (:attr:`arwv_recall`),
    the no-reload per-shot path is used: pre-store every unique waveform once (``WVDT WVNM,<name>``,
    :meth:`store_waveform`) then switch with :meth:`recall_by_name` (``ARWV NAME,<name>``, ~ms) --
    ``ARWV`` restores the waveform's baked ``FREQ`` so one command sets the whole pulse. On older
    firmware (37R6, e.g. the still-un-flashed AWG308) ``ARWV`` is a no-op, so the fallback re-sends
    the full WVDT to the ``active`` slot every shot (:meth:`send_waveform`).
  * **Amplitude is set once** (:meth:`set_amplitude`) in setup -- a per-shot ``BSWV AMP`` would add
    ~400 ms (two ``*OPC?`` round-trips).
  * In DDS mode ``FREQ`` lives INSIDE the WVDT command. ``SRATE MODE,DDS`` is only needed/valid on
    fw < 38R3; 38R3 removed true-arb mode (DDS is the only mode), so that command now returns an
    execution error -- :meth:`connect` only sends it on older firmware.
  * **Same-session readback lag:** ``ARWV?``/``TRIGger:SOURce?`` can return a STALE value right after
    a set in the same VISA session (the set still took effect). Don't gate logic on an immediate
    same-session readback of those.
"""
import logging
import re

logger = logging.getLogger(__name__)

# ARWV NAME recall of user waveforms works on SDG6X firmware >= 6.01.01.38 (37R6 broken, 38R3 fixed).
ARWV_MIN_FW = (6, 1, 1, 38)


class AWGConnection:
    """One Siglent SDG6X channel over USB-VISA (pyvisa). Handle-class semantics."""

    def __init__(self, resource_address, channel):
        self.resource = resource_address
        self.channel = channel
        self.dev = None
        self._rm = None
        self.firmware = None       # parsed fw tuple, e.g. (6, 1, 1, 38); None if unknown
        self.arwv_recall = False   # True iff fw >= ARWV_MIN_FW -> use ARWV NAME (no re-upload)

    def connect(self):
        """Open the USB-VISA handle, clear it, and switch the channel to DDS mode.

        Lazily imports pyvisa (NEEDS-HARDWARE). On a stale-handle failure it recreates the
        ResourceManager once and retries (the pyvisa analog of MATLAB's
        ``instrfindall()/delete`` dance).
        """
        import pyvisa  # lazy: no VISA backend needed to import this module

        try:
            self._rm = pyvisa.ResourceManager()
            self.dev = self._rm.open_resource(str(self.resource))
        except Exception:
            # Stale handle / busy resource -> drop and retry with a fresh manager.
            try:
                if self._rm is not None:
                    self._rm.close()
            except Exception:
                pass
            self._rm = pyvisa.ResourceManager()
            self.dev = self._rm.open_resource(str(self.resource))

        self.dev.timeout = 10000          # ms (MATLAB dev.Timeout = 10 s)
        self.dev.write_termination = "\n"
        self.dev.read_termination = "\n"
        self.dev.write("*CLS")
        idn = self.dev.query("*IDN?").strip()
        self.firmware = self._parse_fw(idn)
        self.arwv_recall = self.firmware is not None and self.firmware >= ARWV_MIN_FW
        logger.info("AWG connected: %s (fw=%s, arwv_recall=%s)",
                    idn, self.firmware, self.arwv_recall)
        # DDS mode: FREQ in the WVDT command controls playback rate. Older fw (< 38R3) needs
        # SRATE MODE,DDS; 38R3 removed true-arb mode -> the command returns an execution error
        # (DDS is the only/default mode), so only send it when it is valid.
        if self.firmware is None or self.firmware < ARWV_MIN_FW:
            self.dev.write("%s:SRATE MODE,DDS" % self.channel)
            self.dev.query("*OPC?")
        return idn

    @staticmethod
    def _parse_fw(idn):
        """Firmware tuple from an *IDN? string ('...,SDG6022X,<sn>,6.01.01.38R3') -> (6,1,1,38).

        Returns None if it can't be parsed (capability then defaults to the safe re-upload path).
        """
        try:
            fw = idn.split(",")[-1].strip()          # '6.01.01.38R3'
            nums = re.findall(r"\d+", fw)             # ['6','01','01','38','3']
            return tuple(int(x) for x in nums[:4]) if len(nums) >= 4 else None
        except Exception:  # noqa: BLE001
            return None

    def disconnect(self):
        try:
            if self.dev is not None:
                self.dev.close()
        except Exception:
            pass
        try:
            if self._rm is not None:
                self._rm.close()
        except Exception:
            pass
        self.dev = None
        self._rm = None

    def build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz, name="active"):
        """Build a DDS-mode WVDT command (``bytes``) for ``binary_data``.

        Mirrors AWGConnection.m exactly: an IEEE-488.2 block header ``#<ndigits><nbytes>``
        followed by the raw big-endian int16 samples. ``AMPL`` must match the ``BSWV AMP`` set
        in :meth:`set_amplitude` so the upload does not override the channel amplitude; ``FREQ``
        is the DDS playback frequency (``1e6 / pulse_width_us``).

        ``name`` is the SDG storage slot: ``"active"`` (the live slot -- the re-upload fallback
        path) or a unique stored name (``"wf_000"`` ... -- the ARWV recall path). ``ARWV NAME``
        restores the baked ``FREQ`` of the stored waveform, so width comes along with the recall.

        Pure (no device access) -- unit-testable without hardware.
        """
        num_bytes = len(binary_data)
        ieee_header = "#%d%d" % (len(str(num_bytes)), num_bytes)
        cmd_prefix = ("%s:WVDT WVNM,%s,WVTP,USER,AMPL,%g,OFST,0,FREQ,%g,WAVEDATA,%s"
                      % (self.channel, name, amplitude_vpp, freq_hz, ieee_header))
        return cmd_prefix.encode("ascii") + bytes(binary_data)

    def send_waveform(self, cmd):
        """Send a pre-built WVDT command (``bytes``) to switch the active waveform (~2 ms).

        The re-upload fallback switch (older firmware): re-sends the full WVDT to the ``active``
        slot. Fire-and-forget (no ``*OPC?``)."""
        self.dev.write_raw(cmd)

    def store_waveform(self, cmd):
        """Upload a pre-built NAMED WVDT to SDG storage (setup-time; waits ``*OPC?``).

        Used once per unique waveform at scan start for the ARWV recall path; the per-shot switch
        is then the cheap :meth:`recall_by_name`."""
        self.dev.write_raw(cmd)
        self.dev.query("*OPC?")

    def set_arb_mode(self):
        """Put the channel in arbitrary-waveform DDS output mode (before ARWV recall / gated burst)."""
        self.dev.write("%s:BSWV WVTP,ARB" % self.channel)
        self.dev.write("%s:ARWV MODE,DDS" % self.channel)
        self.dev.query("*OPC?")

    def recall_by_name(self, name):
        """Switch the active output waveform to a STORED one by name (~ms; fw >= ARWV_MIN_FW).

        No re-upload -- ``ARWV NAME,<name>`` re-points the active waveform (and restores its baked
        ``FREQ``). Fire-and-forget (no ``*OPC?``); the same-session ``ARWV?`` readback may lag."""
        self.dev.write("%s:ARWV NAME,%s" % (self.channel, name))

    def set_amplitude(self, amp_vpp):
        self.dev.write("%s:BSWV AMP,%g" % (self.channel, amp_vpp))
        self.dev.query("*OPC?")
        self.dev.write("%s:BSWV OFST,0.0" % self.channel)
        self.dev.query("*OPC?")

    def configure_burst(self):
        ch = self.channel
        self.dev.write("%s:BTWV STATE,ON" % ch)
        self.dev.write("%s:BTWV GATE_NCYC,GATE" % ch)
        self.dev.write("%s:BTWV TRSR,EXT" % ch)
        self.dev.write("%s:BTWV EDGE,RISE" % ch)
        self.dev.write("%s:BTWV PLRT,POS" % ch)
        err = self.dev.query("SYST:ERR?").strip()
        if "no error" not in err.lower() and not err.startswith("0,"):
            logger.warning("AWGConnection burst config: %s", err)

    def set_frequency(self, freq_hz):
        self.dev.write("%s:BSWV FRQ,%g" % (self.channel, freq_hz))

    def enable_output(self):
        self.dev.write("%s:OUTP ON" % self.channel)
        self.dev.query("*OPC?")

    def disable_output(self):
        """Turn the channel output + burst OFF so the AWG is QUIET (called at scan end / cleanup).

        Without this the AWG is left armed gated -- and a statically-high gate makes it free-run a
        continuous waveform train after the scan."""
        self.dev.write("%s:BTWV STATE,OFF" % self.channel)
        self.dev.write("%s:OUTP OFF" % self.channel)
        self.dev.query("*OPC?")
