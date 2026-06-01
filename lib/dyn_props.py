"""dyn_props.py -- MINIMAL Phase-2 stub of ``matlab_new/lib/DynProps.m``.

Phase 2 (sequence tree & timing) is validated against ``TestExpSeq.m``, which runs
on an **empty** config: no consts, no scan params, and the only access is the
constructor's ``self.C.debug(0)`` fallback read. This stub provides just that --
the ``obj.field(default)`` fallback-call syntax -- so ``ExpSeq`` can be built and
serialized without dragging in the full DynProps/SubProps/ParamPack model.

The full, handle-class, nested, access-tracking port lands in **Phase 3** (config &
globals); see PYTHON_FRONTEND_PLAN.md. Do not build Phase-3 behavior on this stub.
"""


class DynProps:
    def __init__(self, store=None):
        # Bypass our own __setattr__ for the backing store.
        object.__setattr__(self, '_store', dict(store) if store else {})

    def __getattr__(self, name):
        # ``obj.field(default)`` -> stored value if set, else the default.
        # (Phase-3 will return a nested SubProps here instead.)
        store = object.__getattribute__(self, '_store')

        def field(default=None):
            return store.get(name, default)

        return field

    def __setattr__(self, name, value):
        object.__getattribute__(self, '_store')[name] = value
