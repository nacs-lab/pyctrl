"""awg_connection.py -- port of ``matlab_new/YbExptCtrl/sigilentAWG/AWGConnection.m``.

A thin USB-VISA wrapper around one Siglent SDG6X channel, in **DDS mode** (the WVDT ``FREQ``
controls playback rate). MATLAB used ``visadev``; here we use **pyvisa** over the same USB
resource string (e.g. ``USB0::62700::4353::SDG6XFCC900309::0::INSTR``). pyvisa is imported lazily
in :meth:`connect` so this module is import-safe with no VISA backend present (NO-HARDWARE).

SDG6X hard-won rules baked in (see the experiment-running skill / AWG_Integration_Plan.md):
  * **Big-endian int16** waveform bytes (produced by :mod:`gaussian_pulse_waveform`).
  * **ARWV recall is broken** on firmware 6.01.01.37R6 -> we never use ARWV; switching a waveform
    re-sends the full WVDT to the ``active`` slot (``AWGManager`` caches the pre-built command).
  * **Amplitude is set once** (:meth:`set_amplitude`) in setup -- a per-shot ``BSWV AMP`` would add
    ~400 ms (two ``*OPC?`` round-trips).
  * In DDS mode ``FREQ`` lives INSIDE the WVDT command (unlike TrueArb, where ``FREQ`` would clobber
    ``SRATE`` and a ``SRATE VALUE`` must be re-sent after each upload).
"""
import logging

logger = logging.getLogger(__name__)


class AWGConnection:
    """One Siglent SDG6X channel over USB-VISA (pyvisa). Handle-class semantics."""

    def __init__(self, resource_address, channel):
        self.resource = resource_address
        self.channel = channel
        self.dev = None
        self._rm = None

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
        logger.info("AWG connected: %s", idn)
        # DDS mode: FREQ in the WVDT command controls playback rate.
        self.dev.write("%s:SRATE MODE,DDS" % self.channel)
        self.dev.query("*OPC?")
        return idn

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

    def build_waveform_cmd(self, binary_data, amplitude_vpp, freq_hz):
        """Build a DDS-mode WVDT command (``bytes``) for ``binary_data``.

        Mirrors AWGConnection.m exactly: an IEEE-488.2 block header ``#<ndigits><nbytes>``
        followed by the raw big-endian int16 samples. ``AMPL`` must match the ``BSWV AMP`` set
        in :meth:`set_amplitude` so the upload does not override the channel amplitude; ``FREQ``
        is the DDS playback frequency (``1e6 / pulse_width_us``).

        Pure (no device access) -- unit-testable without hardware.
        """
        num_bytes = len(binary_data)
        ieee_header = "#%d%d" % (len(str(num_bytes)), num_bytes)
        cmd_prefix = ("%s:WVDT WVNM,active,WVTP,USER,AMPL,%g,OFST,0,FREQ,%g,WAVEDATA,%s"
                      % (self.channel, amplitude_vpp, freq_hz, ieee_header))
        return cmd_prefix.encode("ascii") + bytes(binary_data)

    def send_waveform(self, cmd):
        """Send a pre-built WVDT command (``bytes``) to switch the active waveform (~2 ms)."""
        self.dev.write_raw(cmd)

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
