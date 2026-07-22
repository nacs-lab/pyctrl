"""RearrangeRnRStep.py -- release-and-recapture step with a t=0 SKIP.

Same physics as ``ReleaseRecaptureStep`` (drop the SLM trap power to release the atoms,
hold for the release time, restore the SLM AOM amplitude to recapture, re-enable the
sample-and-hold; a scope trigger brackets the release window) -- BUT with one behavioural
difference tailored to the rearrange -> R&R survival scan (see YbScans/RearrangeRnRScan.py):

    * ``Time == 0`` => add NOTHING to the sequence and return. No ``AmpSLM`` toggle, no
      ``TTLSampleAndHold`` toggle, no ``TTLScopeTrig`` pulse, no 3 us AOM settle. The trap is
      never dropped, so the shot is a byte-clean "held in traps" baseline -- the not-releasing
      proxy the scan uses as its control point.

Why a NEW step (not a reuse of ReleaseRecaptureStep): the stock ReleaseRecaptureStep does NOT
skip at t=0 -- it still drops ``AmpSLM`` to 0 for a 0 s wait and waits the fixed 3 us AOM
response, so its t=0 shot is a genuine (if brief) release, not a true no-op. This scan wants
t=0 to mean "skip release-and-recapture entirely", so the baseline is exactly "rearrange then
image" with no trap perturbation.

The t=0 test is exact: at serialize() time the swept ``ReleaseRecapture.Time`` DynProp resolves
to a concrete Python ``float`` per scan point (verified: 0.0 at the t=0 point), so
``float(t_release) == 0.0`` is a real numeric compare, and the two branches serialize to
DISTINCT blobs (the t=0 point is ~23 B smaller -- the R&R adds/waits are absent).

Byte note: pulse VALUES (2nd arg to ``add``) are float()-coerced by ``_resolve_pulse``, so bare
ints (``0``/``1``) serialize as ARG_CONST_FLOAT64 -- faithful to MATLAB's doubles. The active
(t>0) branch is byte-identical to ReleaseRecaptureStep.
"""

from consts import Consts


def RearrangeRnRStep(s, g):
    t_release = g.Time(0)

    # t=0 => skip release-and-recapture entirely (no bytes added): the "held in traps" baseline.
    if float(t_release) == 0.0:
        return

    Amp_SLM = g.SLMAOMAmp(Consts().SLM.AOM.Amp)

    s.add('AmpSLM', 0).add('TTLSampleAndHold', 0)
    s.add('TTLScopeTrig', 1)
    s.wait(t_release)
    s.add('TTLScopeTrig', 0)
    s.add('AmpSLM', Amp_SLM)
    s.wait(3e-6)  # wait the response time of our 532 AOM
    s.add('TTLSampleAndHold', 1)
