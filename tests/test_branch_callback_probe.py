"""test_branch_callback_probe.py -- de-risk the SLMRearrangement -> pyctrl port.

SLMRearrangementScan dispatches to RearrangeCommSeq (s1/s2) and RearrangeCommSeq2
(s1/s2/s3). The machinery those lean on is:
  * ``new_basic_seq()``                 -> the s2 / s3 basic sequences,
  * ``cond_branch(True, s2)`` / ``s2.cond_branch(True, s3)``  -> chained branches,
  * ``reg_before_start`` / ``reg_before_bseq`` / ``reg_after_bseq`` /
    ``reg_after_branch`` / ``reg_after_end``  -> the pre/post sequence callbacks
    (pre_run / hand_over_slm / hand_over_slm_2 / post_run).

The byte STRUCTURE of that skeleton is already proven byte-identical to MATLAB for
the real seqs (test_ybseqs_build.py: RearrangeCommSeq + RearrangeCommSeq2). What was
NOT yet exercised end-to-end is the **callback dispatch through the run loop** driven
by a *real* ``cond_branch`` structure: the ported seqs register the callbacks as
``_noop`` (serialize() never runs them), and test_run_seq2.py drives the loop with a
hand-scripted ``next_idxs`` list rather than a sequence actually built with
``cond_branch``. This module closes that gap with a minimal, hardware-free probe:

  1. STRUCTURE/BYTES -- build the probe, decode it, assert the branch layout
     (target_id chain 1->2->3->end) and a byte-exact decode/encode round-trip;
     when the MATLAB capture is present (tools/capture_branch_callback_probe.m),
     assert serialize() is byte-identical to it.
  2. CALLBACKS -- build the SAME probe with RECORDING callbacks, attach a fake
     ``pyseq`` whose ``post_run()`` walks the REAL ``cond_branch`` structure
     (``branch_walk`` below -- not a hand-typed next_idx list), run ``run_real``,
     and assert the callbacks fire in MATLAB's verified order (ExpSeq.m:369-463),
     each one passed the ROOT sequence.

NO-HARDWARE. Config = pyctrl's own executable ``expConfig.py`` via
``SeqConfig.load_real()`` (the same real config the RearrangeComm corpus is
byte-verified against), tick = 1e12 (production rate). The libnacs engine is never
loaded: ``run_real`` runs against the fake pyseq, ``serialize()`` is pure.
"""

import json
import os

import pytest

import compare_bytes
import run_seq2
import seq_manager
from conftest import _TESTS_DIR
from exp_seq import ExpSeq
from seq_config import SeqConfig

pytestmark = pytest.mark.no_hardware

_REF = os.path.join(_TESTS_DIR, "reference_branch_probe", "probe_reference.json")
_PROBE_NAME = "BranchCallbackProbe"


# --------------------------------------------------------------------------- #
# The probe builder -- mirror of tools/capture_branch_callback_probe.m
# --------------------------------------------------------------------------- #
def build_probe(s, log=None):
    """Build the minimal 3-bseq branch+callback probe on a fresh ExpSeq ``s``.

    Mirrors RearrangeCommSeq2's skeleton (root -> s2 -> s3 via ``new_basic_seq`` +
    ``cond_branch(True, ...)``) but strips ALL hardware: each bseq does one trivial
    synthetic TTL step, and every callback is a recorder (or a no-op when ``log`` is
    None, used for the byte capture -- callbacks have zero byte impact either way).
    The full per-bseq callback triple is registered on EVERY bseq (including the
    root) so the dispatch of all five hook types is exercised. Returns ``s``.
    """
    root = s

    s.reg_before_start(_rec(log, "before_start", root))
    _reg_bseq_cbs(s, "bs1", log, root)
    s.add_step(1).add("Device1/CH1", 1)

    s2 = s.new_basic_seq()
    s.cond_branch(True, s2)
    _reg_bseq_cbs(s2, "bs2", log, root)
    s2.add_step(1).add("Device1/CH1", 0)

    s3 = s.new_basic_seq()
    s2.cond_branch(True, s3)
    _reg_bseq_cbs(s3, "bs3", log, root)
    s3.add_step(1).add("Device1/CH1", 1)

    s.reg_after_end(_rec(log, "after_end", root))
    return s


def _reg_bseq_cbs(bseq, tag, log, root):
    bseq.reg_before_bseq(_rec(log, "before_bseq:%s" % tag, root))
    bseq.reg_after_bseq(_rec(log, "after_bseq:%s" % tag, root))
    bseq.reg_after_branch(_rec(log, "after_branch:%s" % tag, root))


def _rec(log, tag, root):
    """A callback that records (tag, received-the-root-seq) -- or a no-op if log is None."""
    def _fn(seq):
        if log is not None:
            log.append((tag, seq is root))
    return _fn


# --------------------------------------------------------------------------- #
# Branch walk -- replicate the engine's routing from the REAL cond_branch tree
# --------------------------------------------------------------------------- #
def branch_walk(seq, max_steps=64):
    """Walk the basic-sequence branch graph the way the engine's ``post_run`` does.

    Every probe branch cond is the literal ``True`` (like RearrangeCommSeq's
    ``cond_branch(True, ...)``), so the first branch is always taken; a bseq with no
    matching branch falls through to its ``default_target`` (None -> end -> idx 0).
    Reads the REAL ``branches`` / ``default_target`` built by ``cond_branch`` -- so the
    next_idx sequence fed to the fake pyseq is derived from the structure under test,
    not hand-typed. Returns ``(visited_idxs, next_idxs)``.
    """
    visited, nexts = [], []
    idx = 1
    for _ in range(max_steps):
        if idx == 0:
            break
        visited.append(idx)
        bseq = seq if idx == 1 else seq.basic_seqs[idx - 2]
        nxt = _next_idx(bseq)
        nexts.append(nxt)
        idx = nxt
    else:
        raise AssertionError("branch walk did not terminate (cycle in cond_branch?)")
    return visited, nexts


def _next_idx(bseq):
    for br in bseq.branches:
        if _cond_true(br["cond"]):
            return 0 if br["target"] is None else br["target"].bseq_id
    return 0 if bseq.default_target is None else bseq.default_target.bseq_id


def _cond_true(cond):
    # The probe uses literal-bool conds (mirrors RearrangeCommSeq's cond_branch(True,
    # ...)). A SeqVal cond would need the real engine to evaluate -- out of scope here.
    if isinstance(cond, bool):
        return cond
    raise AssertionError(
        "branch_walk only handles literal-bool conds (got %r); a SeqVal cond needs "
        "the real engine" % (cond,))


# --------------------------------------------------------------------------- #
# Fake pyseq -- post_run() returns the precomputed walk; every call is logged
# --------------------------------------------------------------------------- #
class _ProbePyseq:
    """A generated-sequence stand-in for the run loop (no engine, no hardware).

    ``post_run()`` returns the next bseq index from the route ``branch_walk`` derived
    from the real ``cond_branch`` structure. Every engine call is appended to the
    shared log so its interleaving with the recording callbacks can be asserted
    exactly against MATLAB's run_bseq / run_real order.
    """

    def __init__(self, log, next_idxs):
        self._log = log
        self._next = list(next_idxs)

    def init_run(self):
        self._log.append("init_run")

    def pre_run(self):
        self._log.append("pre_run")

    def start(self):
        self._log.append("start")

    def wait(self, timeout_ms):
        self._log.append("wait")
        return True

    def cur_bseq_length(self):
        self._log.append("cur_bseq_length")
        return 0

    def post_run(self):
        self._log.append("post_run")
        return self._next.pop(0)


# --------------------------------------------------------------------------- #
# Fixture -- pyctrl's own expConfig.py (real config), production tick rate
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_config():
    """Real expConfig.py + production tick; reset both in teardown (process singletons)."""
    SeqConfig.reset()
    SeqConfig.load_real()
    seq_manager.override_tick_per_sec(1e12)
    yield
    seq_manager.override_tick_per_sec(0)
    SeqConfig.reset()


# --------------------------------------------------------------------------- #
# 1. STRUCTURE / BYTES
# --------------------------------------------------------------------------- #
def test_probe_branch_structure(real_config):
    """The s1/s2/s3 + cond_branch skeleton serializes to the expected branch layout
    and re-encodes byte-identically (the decoder is the byte gate)."""
    s = build_probe(ExpSeq())
    got = s.serialize()
    seq = compare_bytes.decode(got)

    assert len(seq["basicseqs"]) == 3
    b0, b1, b2 = seq["basicseqs"]
    # root branches to bseq 2; bseq 2 branches to bseq 3; bseq 3 falls through to end.
    assert len(b0["branches"]) == 1 and b0["branches"][0]["target_id"] == 2
    assert len(b1["branches"]) == 1 and b1["branches"][0]["target_id"] == 3
    assert len(b2["branches"]) == 0 and b2["default_target"] == 0
    assert b0["default_target"] == 0 and b1["default_target"] == 0

    # decode -> encode round-trips byte-for-byte.
    assert compare_bytes.encode(seq) == got
    # repeatable: a fresh build serializes identically.
    assert build_probe(ExpSeq()).serialize() == got


def test_branch_walk_routes_through_s2_s3(real_config):
    """The branch route derived from the real cond_branch structure is 1 -> 2 -> 3 -> end."""
    s = build_probe(ExpSeq())
    visited, nexts = branch_walk(s)
    assert visited == [1, 2, 3]
    assert nexts == [2, 3, 0]


def _ref_bytes():
    if not os.path.exists(_REF):
        return None
    with open(_REF) as f:
        for e in json.load(f):
            if e.get("name") == _PROBE_NAME and e.get("status") == "ok":
                return bytes.fromhex(e["bytes"])
    return None


@pytest.mark.skipif(
    _ref_bytes() is None,
    reason="no MATLAB capture (run tools/capture_branch_callback_probe.m)")
def test_probe_bytes_match_matlab(real_config):
    """serialize() is byte-identical to the MATLAB capture of the same probe."""
    want = _ref_bytes()
    got = build_probe(ExpSeq()).serialize()
    if got != want:
        d = compare_bytes.diff(compare_bytes.decode(got), compare_bytes.decode(want))
        raise AssertionError(
            "%s: %d bytes vs MATLAB %d; first diff at %s"
            % (_PROBE_NAME, len(got), len(want), d))


# --------------------------------------------------------------------------- #
# 2. CALLBACKS through the run loop (the gap this module closes)
# --------------------------------------------------------------------------- #
def _bseq_block(tag):
    return [
        ("before_bseq:%s" % tag, True), "pre_run", "start", "wait",
        ("after_bseq:%s" % tag, True), "cur_bseq_length", "post_run",
        ("after_branch:%s" % tag, True),
    ]


def test_probe_callback_firing_order(real_config):
    """Build the probe with recording callbacks, run it through ``run_real`` with the
    branch-walking fake pyseq, and assert the EXACT interleaved order of callbacks and
    engine calls matches MATLAB's run_real / run_bseq (ExpSeq.m:369-463). Each callback
    must receive the ROOT sequence."""
    log = []
    s = build_probe(ExpSeq(), log)

    # next_idxs come from the real cond_branch structure, not a hand-typed list.
    visited, nexts = branch_walk(s)
    assert visited == [1, 2, 3]
    s.pyseq = _ProbePyseq(log, nexts)

    # clock/sleep injected so the tail wall-clock pause never actually sleeps.
    run_seq2.run_real(s, clock=lambda: 0.0, sleep=lambda dt: None)

    expected = (
        [("before_start", True), "init_run"]
        + _bseq_block("bs1") + _bseq_block("bs2") + _bseq_block("bs3")
        + [("after_end", True)]
    )
    assert log == expected

    # explicit, in case the big-list assertion ever loosens: every callback got the root.
    cb_events = [e for e in log if isinstance(e, tuple)]
    assert cb_events and all(is_root for (_tag, is_root) in cb_events), \
        "a callback received the wrong seq argument"


def test_callbacks_have_zero_byte_impact(real_config):
    """Registering recording callbacks does not change the serialized bytes (they run
    only at run time, never in serialize()) -- so the byte capture (no-op callbacks)
    and the callback-firing build (recording callbacks) describe the same sequence."""
    with_cbs = build_probe(ExpSeq(), log=[]).serialize()
    without = build_probe(ExpSeq(), log=None).serialize()
    assert with_cbs == without
