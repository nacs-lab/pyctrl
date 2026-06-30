"""StrobeImag399Step.py -- INTERLEAVED strobe imaging with BEAM ALTERNATION, pyctrl.

NOT a byte-transliteration of ``matlab_new/YbSteps/StrobeImag399Step.m`` (that MATLAB "strobe" step
never strobed). NET-NEW behavior (no byte reference -> THE ONE RULE byte-equality does not apply),
after the Yb stroboscopic imaging scheme in arXiv:2507.01011 -- adapted to OUR regime.

``g = s.C.Imag399``. Per super-cycle (THREE phases):
  1. BEAM 1 pulse (``Strobe.BeamPulseTime``, default 1 us): AmpAbsImag (img1 beam) ON, Amp399Imag2 OFF,
     556 OFF.
  2. BEAM 2 pulse (``Strobe.BeamPulseTime``): Amp399Imag2 (img2 beam) ON, AmpAbsImag OFF, 556 OFF.
  3. RECOOL window (``Strobe.RecoolTime``): both 399 OFF, 556 X+h ON at the SEPARATE strobe-recool
     tones (``Strobe.Cool556.{X,h}``).
The two 399 imaging beams are COUNTER-PROPAGATING, so alternating them (not both-on) cancels the net
photon recoil over a pair -- the paper's "400 ns pulses short enough to mitigate momentum transfer
from a single beam", in our 1 us regime. period = 2*BeamPulseTime + RecoolTime.

n_cycles = round(Orca.ExposureTime / period): the train auto-fills the camera frame (external-trigger
camera, one Orca rising edge = one frame of fixed Orca.ExposureTime). Choose the collect window via
Orca.ExposureTime; both the camera and this train follow it.

SCANNING the period: BeamPulseTime/RecoolTime set the Python loop count -> they are FIXED scalars,
NEVER a .scan() axis. Sweep them across ROUNDS (tools/strobe_imaging_round.py --pulse-time), an outer
loop -- one submission per period.

NO scope-sync TTL is wired -- trigger the scope on the FIRST 399 PD pulse (tools/strobe_scope_capture.py).
HARDWARE REALITY: AOM optical rise ~1 us, DDS ~500 ns. At BeamPulseTime=1 us the AOM barely reaches
full intensity and the OFF edge bleeds into the next phase -> partial-overlap sawtooth, NOT clean
squares. The realized per-beam on-fraction is what the scope measures.

WARN -- sequence size: n_cycles ~ Orca.ExposureTime/period. Small RecoolTime -> many cycles -> LARGE
per-shot sequence (compile time / FPGA instruction memory). Validate with a short Orca.ExposureTime
and/or a bigger RecoolTime first. Pulse VALUES are float()-coerced; 0.0/1.0 verbatim.
"""

from consts import Consts


def StrobeImag399Step(s, g):
    t_pulse = g.Strobe.BeamPulseTime(Consts().Imag399.Strobe.BeamPulseTime)
    t_recool = g.Strobe.RecoolTime(Consts().Imag399.Strobe.RecoolTime)
    n_burst = max(1, int(g.Strobe.PulsesPerBurst(Consts().Imag399.Strobe.PulsesPerBurst)))
    t_exposure = Consts().Orca.ExposureTime           # the camera frame == the strobe train
    period = 2.0 * n_burst * t_pulse + t_recool       # n_burst beam1/beam2 pairs + one recool
    n_cycles = max(1, int(round(t_exposure / period))) if period > 0 else 1

    Freq_Imag399 = Consts().Resonance399Freq + g.FreqDetuning(Consts().Imag399.FreqDetuning)
    Amp_Imag399_1 = g.Amp1(Consts().Imag399.Amp1)   # beam 1 -> AmpAbsImag   (img1)
    Amp_Imag399_2 = g.Amp2(Consts().Imag399.Amp2)   # beam 2 -> Amp399Imag2  (img2)

    # SEPARATE strobe-recool 556 tones (not Imag399.Cool556).
    reson556 = Consts().Resonance556mj0Freq
    Freq_Recool556X = reson556 + g.Strobe.Cool556.X.FreqDetuning(Consts().Imag399.Strobe.Cool556.X.FreqDetuning)
    Freq_Recool556h = reson556 + g.Strobe.Cool556.h.FreqDetuning(Consts().Imag399.Strobe.Cool556.h.FreqDetuning)
    Amp_Recool556X = g.Strobe.Cool556.X.Amp(Consts().Imag399.Strobe.Cool556.X.Amp)
    Amp_Recool556h = g.Strobe.Cool556.h.Amp(Consts().Imag399.Strobe.Cool556.h.Amp)

    # Shutters open ONCE (mechanical -- can't switch at us); set freqs ONCE (only amps toggle).
    s.add('TTL399AbsImagShutter', 1)
    s.add('TTL399Imag2Shutter', 1)
    s.add('TTL556RydbergShutter', 0)
    s.add('FreqAbsImag', Freq_Imag399).add('Freq399Imag2', Freq_Imag399)
    s.add('Freq556MOTX', Freq_Recool556X).add('Freq556RydbergMOTh', Freq_Recool556h)

    s.wait(3e-3)  # wait for the shutter

    # One Orca trigger -> one frame; the strobe train runs inside Orca.ExposureTime.
    s.add_step(100e-6).add('TTLOrcaTrig', 1)
    s.add('TTLOrcaTrig', 0)

    for _ in range(n_cycles):
        # IMAGE BURST: n_burst alternating beam1/beam2 SHORT pulses (556 off the whole burst).
        s.add('Amp556MOTX', 0.0).add('Amp556RydbergMOTh', 0.0)
        for _ in range(n_burst):
            s.add('AmpAbsImag', Amp_Imag399_1).add('Amp399Imag2', 0.0)   # beam 1 (img1)
            s.wait(t_pulse)
            s.add('AmpAbsImag', 0.0).add('Amp399Imag2', Amp_Imag399_2)   # beam 2 (img2), cancels recoil
            s.wait(t_pulse)
        # RECOOL: 399 off, 556 on
        s.add('AmpAbsImag', 0.0).add('Amp399Imag2', 0.0)
        s.add('Amp556MOTX', Amp_Recool556X).add('Amp556RydbergMOTh', Amp_Recool556h)
        s.wait(t_recool)

    # Train done: everything off, reset shutters.
    s.add('AmpAbsImag', 0.0).add('Amp399Imag2', 0.0)
    s.add('Amp556MOTX', 0.0).add('Amp556RydbergMOTh', 0.0)
    s.add('TTL399AbsImagShutter', 0)
    s.add('TTL399Imag2Shutter', 0)
    s.wait(3e-3)  # wait for the shutter
