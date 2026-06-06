"""Auto-dump core (engine-free): SeqDumpSession + the run_seq on_compile hook.

The byte production (engine get_nominal_output) is injected as a fake, so this
exercises the dedup, file naming, manifest schema, and the run-loop wiring
WITHOUT the libnacs engine or hardware.

    python -m pytest pyctrl/tests/test_seq_dump.py -v
"""

import json
import os
import types

import seq_dump
import run_seq


# --------------------------------------------------------------------------- #
# Minimal fakes.
# --------------------------------------------------------------------------- #
class FakeScanGroup:
    """nseq points; each maps to a seqid + a scanned value (group-1, 1 axis)."""

    def __init__(self, seqids, values, path="Pushout.Time"):
        self._seqids = seqids          # per-point (1-based) compiled seqid
        self._values = values          # per-point scanned value
        self._path = path
        self._axis_values = sorted(set(values))

    def nseq(self):
        return len(self._seqids)

    def getseq_with_var(self, n):
        seqid = self._seqids[n - 1]
        seqparam = {"Pushout": {"Time": self._values[n - 1]}}
        return seqid, seqparam, []

    def scandim(self, idx):
        return 1

    def axisnum(self, idx, dim):
        return 1 if dim == 1 else 0

    def get_scanaxis(self, idx, dim, field):
        return self._axis_values, self._path


class _GG:
    """DynProps-ish: supports both ``g.x = v`` and ``g.x(default)`` reads."""

    def __init__(self):
        object.__setattr__(self, "_d", {})

    def __setattr__(self, k, v):
        self._d[k] = v

    def __getattr__(self, k):
        val = self._d.get(k, 0)
        return lambda default=None, _v=val: _v


class FakeSeqConfig:
    def __init__(self):
        self.G = _GG()


def _fake_seq(tag):
    # after_end_cbs / reg_after_end mirror the ExpSeq run hooks: the dump is armed
    # on after_end (run_seq._arm_dump) and fired by run_real, not at compile time.
    seq = types.SimpleNamespace(C=types.SimpleNamespace(RESTART=lambda d=0: 0), tag=tag)
    seq.after_end_cbs = []
    seq.reg_after_end = seq.after_end_cbs.append
    return seq


def _run_real_firing_after_end(seq):
    """Fake run_real that fires the seq's after_end callbacks (where the dump is armed)."""
    for cb in getattr(seq, "after_end_cbs", []):
        cb(seq)


def _fake_seq_with_globals(tag, globals_map):
    """A fake ExpSeq exposing the runtime-global API the capture reads.

    ``globals_map`` is {global_id: injected_value}; ``globals`` mirrors ExpSeq's list of
    {'id','persist','init_val'} dicts and ``get_global(id)`` returns the injected value.
    """
    seq = _fake_seq(tag)
    seq.globals = [{"id": gid, "persist": True, "init_val": 0.0}
                   for gid in globals_map]
    seq.get_global = lambda gid, _m=globals_map: _m[int(gid)]
    return seq


# --------------------------------------------------------------------------- #
# SeqDumpSession
# --------------------------------------------------------------------------- #
def test_session_dedup_and_manifest(tmp_path):
    sg = FakeScanGroup([10, 11, 10, 11], [1e-3, 2e-3, 1e-3, 2e-3])
    seq_dir = str(tmp_path / "sequence")

    def fake_dump(seq, name):
        assert name == "RydDetSeq"            # no datetime stamp -> bare name
        return ("SEQ:%s" % seq.tag).encode()

    sess = seq_dump.SeqDumpSession(seq_dir, sg, scan_id="20250619170317",
                                   seq_name="RydDetSeq", dump_fn=fake_dump)

    # The run loop fires on_compile once per unique seqid; a repeat is a no-op.
    sess.on_compile(1, 10, _fake_seq("A"))
    sess.on_compile(2, 11, _fake_seq("B"))
    sess.on_compile(3, 10, _fake_seq("A"))    # dedup -> no new file

    files = sorted(f for f in os.listdir(seq_dir) if f.endswith(".seq"))
    assert files == ["point_00001__seqid_10.seq", "point_00002__seqid_11.seq"]
    assert len(sess.unique) == 2

    man = sess.finalize()
    assert man is not None
    on_disk = json.load(open(os.path.join(seq_dir, "manifest.json")))
    assert on_disk["scan_id"] == "20250619170317"
    assert on_disk["seq"] == "RydDetSeq"
    assert [p["n"] for p in on_disk["points"]] == [1, 2, 3, 4]
    # every point references the .seq its seqid produced
    assert on_disk["points"][0]["file"] == "point_00001__seqid_10.seq"
    assert on_disk["points"][2]["file"] == "point_00001__seqid_10.seq"   # reused
    assert on_disk["points"][1]["file"] == "point_00002__seqid_11.seq"
    # per-point scanned value pulled straight from getseq_with_var's seqparam
    assert on_disk["points"][0]["scanned"]["Pushout.Time"] == 1e-3
    assert on_disk["points"][1]["scanned"]["Pushout.Time"] == 2e-3
    assert on_disk["scanned_axes"][0]["path"] == "Pushout.Time"
    assert on_disk["scanned_axes"][0]["values"] == [1e-3, 2e-3]
    assert on_disk["unique_seqs"]["10"] == "point_00001__seqid_10.seq"


def test_session_dump_failure_never_raises(tmp_path):
    sg = FakeScanGroup([10], [1e-3])

    def boom(seq, name):
        raise RuntimeError("engine exploded")

    sess = seq_dump.SeqDumpSession(str(tmp_path / "sequence"), sg, dump_fn=boom)
    sess.on_compile(1, 10, _fake_seq("A"))     # must not raise
    assert sess.unique == {}
    # finalize still writes a manifest (points present, file=None for the seqid)
    man = sess.finalize()
    assert man["points"][0]["file"] is None


# --------------------------------------------------------------------------- #
# run_seq.run_scan_group on_compile wiring
# --------------------------------------------------------------------------- #
def test_run_scan_group_calls_on_compile_once_per_seqid():
    sg = FakeScanGroup([10, 11, 10, 11], [1e-3, 2e-3, 1e-3, 2e-3])
    calls = []

    def on_compile(arg0, seqid, seq):
        calls.append((arg0, seqid, seq.tag))

    def compile_point(seqfn, seqparam):
        # tag the seq by its scanned value so we can see which point compiled it
        return _fake_seq("v=%s" % seqparam["Pushout"]["Time"])

    # run_real fires after_end (where _arm_dump registered the dump). Reused seqids
    # (points 3,4) re-run the SAME seq object whose dump cb already fired -> one-shot.
    res = run_seq.run_scan_group(
        seqfn=lambda s: None, scangroup=sg,
        compile_point=compile_point, run_real=_run_real_firing_after_end,
        seq_config=FakeSeqConfig(), control=None, on_compile=on_compile)

    assert res["status"] == "ok"
    assert res["nseq"] == 4                      # all four shots ran
    # on_compile fired once per UNIQUE seqid (10 and 11), not per shot
    assert len(calls) == 2
    assert {c[1] for c in calls} == {10, 11}
    # first arg0 that compiled each seqid
    by_seqid = {c[1]: c[0] for c in calls}
    assert by_seqid == {10: 1, 11: 2}


def test_run_scan_group_dump_not_fired_at_compile():
    """The dump is ARMED at compile but FIRED only when run_real runs after_end --
    a run_real that ignores after_end (or never runs) yields no dump."""
    sg = FakeScanGroup([10, 11], [1e-3, 2e-3])
    calls = []

    res = run_seq.run_scan_group(
        seqfn=lambda s: None, scangroup=sg,
        compile_point=lambda f, p: _fake_seq("x"),
        run_real=lambda seq: None,                # never fires after_end_cbs
        seq_config=FakeSeqConfig(), control=None,
        on_compile=lambda a, s, q: calls.append(s))

    assert res["nseq"] == 2
    assert calls == []                            # nothing dumped at compile time


def test_run_scan_group_no_on_compile_is_fine():
    sg = FakeScanGroup([10, 10], [1e-3, 1e-3])
    res = run_seq.run_scan_group(
        seqfn=lambda s: None, scangroup=sg,
        compile_point=lambda f, p: _fake_seq("x"), run_real=lambda seq: None,
        seq_config=FakeSeqConfig(), control=None)
    assert res["status"] == "ok"
    assert res["nseq"] == 2


# --------------------------------------------------------------------------- #
# GlobalsCaptureSession (Q-F: UNGATED runtime-global capture)
# --------------------------------------------------------------------------- #
def test_globals_capture_dedup_and_write(tmp_path):
    seq_dir = str(tmp_path / "sequence")
    sess = seq_dump.GlobalsCaptureSession(seq_dir, scan_id="20250619170317",
                                          seq_name="RydDetSeq")
    sess.on_globals(1, 10, _fake_seq_with_globals("A", {0: 1.234e8, 1: 5.0}))
    sess.on_globals(2, 11, _fake_seq_with_globals("B", {0: 9.99e7}))
    sess.on_globals(3, 10, _fake_seq_with_globals("A2", {0: 0.0}))   # dedup -> ignored

    doc = sess.finalize()
    assert doc is not None
    on_disk = json.load(open(os.path.join(seq_dir, "globals.json")))
    assert on_disk["scan_id"] == "20250619170317"
    assert on_disk["seq"] == "RydDetSeq"
    assert set(on_disk["globals"].keys()) == {"10", "11"}
    # captured value == what get_global returned (the injected runtime value)
    g10 = {e["id"]: e["value"] for e in on_disk["globals"]["10"]}
    assert g10 == {0: 1.234e8, 1: 5.0}
    # seqid 10's repeat (value 0.0) was deduped, NOT overwritten
    g11 = {e["id"]: e["value"] for e in on_disk["globals"]["11"]}
    assert g11 == {0: 9.99e7}
    # per-entry metadata is carried for diagnostics / reconstruction
    assert on_disk["globals"]["10"][0]["persist"] is True


def test_globals_capture_skips_when_no_globals(tmp_path):
    """A seq with no runtime globals must not litter an otherwise-absent sequence/ dir."""
    seq_dir = str(tmp_path / "sequence")
    sess = seq_dump.GlobalsCaptureSession(seq_dir)
    sess.on_globals(1, 10, _fake_seq_with_globals("A", {}))   # no globals
    assert sess.finalize() is None
    assert not os.path.exists(seq_dir)


def test_globals_capture_failure_never_raises(tmp_path):
    seq_dir = str(tmp_path / "sequence")
    sess = seq_dump.GlobalsCaptureSession(seq_dir)
    seq = _fake_seq("A")
    seq.globals = [{"id": 0, "persist": False, "init_val": 0.0}]

    def boom(_gid):
        raise RuntimeError("engine gone")

    seq.get_global = boom
    sess.on_globals(1, 10, seq)            # must not raise
    assert sess.by_seqid == {"10": []}     # the bad global was skipped
    assert sess.finalize() is None         # nothing to write
    assert not os.path.exists(seq_dir)


def test_run_scan_group_calls_on_globals_once_per_seqid():
    """on_globals is armed in after_end (UNGATED -- here on_compile is None) and fires
    once per UNIQUE seqid, like the dump hook."""
    sg = FakeScanGroup([10, 11, 10, 11], [1e-3, 2e-3, 1e-3, 2e-3])
    calls = []

    def on_globals(arg0, seqid, seq):
        calls.append((arg0, seqid, seq.tag))

    def compile_point(seqfn, seqparam):
        return _fake_seq_with_globals("v=%s" % seqparam["Pushout"]["Time"], {0: 1.0})

    res = run_seq.run_scan_group(
        seqfn=lambda s: None, scangroup=sg,
        compile_point=compile_point, run_real=_run_real_firing_after_end,
        seq_config=FakeSeqConfig(), control=None, on_globals=on_globals)

    assert res["status"] == "ok"
    assert res["nseq"] == 4
    assert len(calls) == 2                       # once per unique seqid, not per shot
    assert {c[1] for c in calls} == {10, 11}


def test_run_scan_group_on_globals_not_fired_at_compile():
    """Like the dump, the capture is ARMED at compile but FIRED only in after_end."""
    sg = FakeScanGroup([10, 11], [1e-3, 2e-3])
    calls = []
    res = run_seq.run_scan_group(
        seqfn=lambda s: None, scangroup=sg,
        compile_point=lambda f, p: _fake_seq_with_globals("x", {0: 1.0}),
        run_real=lambda seq: None,               # never fires after_end_cbs
        seq_config=FakeSeqConfig(), control=None,
        on_globals=lambda a, s, q: calls.append(s))
    assert res["nseq"] == 2
    assert calls == []                           # nothing captured at compile time
