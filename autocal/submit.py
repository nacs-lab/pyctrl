"""autocal.submit -- build + submit a background (calibration) scan for a (cal, pattern).

The submit is purely additive on top of the existing descriptor path: it sets ``background=True``
(so the ExptServer background lane runs it only when idle and yields it to foreground instantly)
and ``cycle=True`` (so it round-robins). The scan's run controls get a finite ``rep`` so it
self-stops -- which is what lets ``cycle`` re-queue it (a forever scan never "finishes" to cycle).

Identity is carried in the descriptor ``label`` as ``autocal::<cal_key>::<pattern>`` so the
controller can map a completed job in the queue history back to which (pattern, cal) it fed.

Self-asserting pattern: when the controller is auto-cycling to a NON-home pattern it passes
``loading_phase`` (+ ``loading_defocus``) so the scan writes that pattern's hologram itself
(upholding "a foreground scan always finds home" -- a non-home pattern is only ever on the SLM for
the duration of one short cal). For the home/active pattern we pass no ``loading_phase`` (matches
the current spectroscopy behaviour -- no extra SLM write).

The descriptor-building half is pure (injected ``build_fn`` / ``to_descriptor`` in tests); only
``submit_background`` touches the network, and only when called with a real ``url``.
"""

import json

LABEL_PREFIX = "autocal"
LABEL_SEP = "::"


def make_label(cal_key, pattern):
    return LABEL_SEP.join([LABEL_PREFIX, cal_key, pattern])


def parse_label(label):
    """``autocal::<cal_key>::<pattern>`` -> ``(cal_key, pattern)``; ``None`` if not an autocal label.
    The pattern may itself contain ``::``-free text; we split into exactly 3 from the left."""
    if not label or not str(label).startswith(LABEL_PREFIX + LABEL_SEP):
        return None
    parts = str(label).split(LABEL_SEP, 2)
    if len(parts) != 3 or parts[0] != LABEL_PREFIX:
        return None
    return parts[1], parts[2]


def make_description(cal_def, pattern, *, requires_switch=False):
    base = cal_def.get("desc", cal_def.get("label", "calibration"))
    extra = (" Auto-cycled onto pattern '%s' (self-asserts its hologram, restores home after)."
             % pattern) if requires_switch else (" Background calibration on pattern '%s'." % pattern)
    return ("[auto-calibration] %s%s Continuous background lane; pooled across yielded partial "
            "runs until target shots, then fit." % (base, extra))


def build_descriptor(cal_key, cal_def, pattern, *, reps, loading_phase=None, loading_defocus=None,
                     cycle=True, requires_switch=False, build_fn=None, to_descriptor=None,
                     extra_runp=None):
    """Build the background descriptor dict for one cal scan. Pure (deps injected).

    ``build_fn(**build_kwargs) -> ScanGroup`` defaults to importing ``cal_def['module']``'s
    ``build``. ``to_descriptor`` defaults to ``scan_export.scangroup_to_descriptor``."""
    if build_fn is None:
        build_fn = _import_build(cal_def["module"])
    if to_descriptor is None:
        from scan_export import scangroup_to_descriptor as to_descriptor  # type: ignore

    g = build_fn(**(cal_def.get("build_kwargs") or {}))
    rp = g.runp()
    # Self-assert the hologram only when switching to a non-home pattern.
    if loading_phase is not None:
        rp.loading_phase = loading_phase
        if loading_defocus is not None:
            rp.loading_defocus = loading_defocus
    for k, v in (extra_runp or {}).items():
        setattr(rp, k, v)

    opts = {"rep": int(reps)}
    label = make_label(cal_key, pattern)
    desc = to_descriptor(g, cal_def["seq"], opts=opts, label=label,
                         description=make_description(cal_def, pattern,
                                                      requires_switch=requires_switch),
                         background=True, cycle=cycle)
    return desc


def submit_background(desc, url, *, submit_fn=None, label=None):
    """Submit a built background descriptor to the running backend. IO -- live phases only.

    ``submit_fn(url, desc_json, label) -> id`` defaults to ``yb_start_scan.submit_descriptor``."""
    if submit_fn is None:
        from yb_start_scan import submit_descriptor as submit_fn  # type: ignore
    desc_json = json.dumps(desc, ensure_ascii=False)
    return submit_fn(url, desc_json, label or desc.get("label"))


# --------------------------------------------------------------------------- #
def _import_build(module_name):
    import importlib
    mod = importlib.import_module(module_name)
    if not hasattr(mod, "build"):
        raise AttributeError("scan module %r has no build()" % module_name)
    return mod.build
