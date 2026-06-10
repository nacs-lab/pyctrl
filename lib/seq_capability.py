"""seq_capability.py -- declarative run-loop capability flags for a sequence.

A seq function declares run-loop capabilities at its definition; the runner reads them
**pre-compile** (when only the resolved seq function is available, before ``run_scan_group``
lazily builds it). These are run-loop metadata only -- never serialized, so THE ONE RULE is
untouched.

Currently one flag:

    owns_frames -- the seq grabs + stores its OWN camera frames mid-sequence (rearrangement reads
                   img1 at a handoff to decide the move). When True, the runner does NOT add the
                   shared post-shot capture ``post_cb`` -- the seq publishes frames itself (via the
                   ExptServer persister). Default (undeclared) is False -> the shared capture
                   applies, which is every normal imaging scan.

Usage::

    from seq_capability import seq_capabilities

    @seq_capabilities(owns_frames=True)
    def RearrangeCommSeq(s):
        ...

The decorator returns the function UNCHANGED (it only stamps attributes), so dispatch's
``getattr(mod, name)`` and the ``seqfn(s)`` build call are unaffected. The runner reads a flag
with :func:`has_capability` (default False), so a seq that declares nothing behaves exactly as
before.
"""


def seq_capabilities(*, owns_frames=False):
    """Decorator: stamp run-loop capability flags onto a seq function (see module docstring)."""
    def deco(fn):
        fn.owns_frames = bool(owns_frames)
        return fn
    return deco


def has_capability(seq, name, default=False):
    """Read a capability flag off a seq function (or anything), defaulting when undeclared."""
    return getattr(seq, name, default)
