"""capture_config_reference.py -- regenerate the config drift-oracle snapshot FROM
``expConfig.py`` (pyctrl is the config source of truth; NO MATLAB needed).

Since 2026-06-05, ``pyctrl/expConfig.py`` is the human-edited source of truth for the
front-end config (the gradual MATLAB -> pyctrl switch). The committed snapshot
``tests/reference/config_reference.json`` is a frozen copy of ``expConfig.build_config()``
that the drift oracle (``tests/test_exp_config.py``) checks ``expConfig.py`` against -- so an
accidental/unintended edit to ``expConfig.py`` fails loudly until this is re-run on purpose.

``build_config()`` already emits exactly the snapshot schema (``channel_alias_keys`` /
``channel_alias_vals`` / ``consts`` / ``default_vals_*`` / ``ni_*``), so the snapshot is just
its JSON dump -- the regeneration is exact and engine-free (imports only ``expConfig``; no
``SeqConfig`` / ``SeqManager`` / ``Manager``).

Supersedes ``tools/capture_config_reference.m`` (which sourced ``matlab_new/expConfig.m``).
The MATLAB capture remains usable for a transition parity check, but ``expConfig.py`` is now
authoritative. NOTE: this regenerates ONLY the config oracle -- the byte oracles
(``ybseqs_reference.json`` / ``scan_point_reference.json``) enforce THE ONE RULE (byte-equality
vs MATLAB) and stay MATLAB-captured; re-capture those from MATLAB after a recalibration.

Run:
    python pyctrl/tools/capture_config_reference.py
    python pyctrl/tools/capture_config_reference.py <out.json>
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)            # pyctrl/
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import expConfig


def main(out_json=None):
    if out_json is None:
        out_json = os.path.join(_ROOT, "tests", "reference", "config_reference.json")
    cfg = expConfig.build_config()
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    # Compact separators to match the prior MATLAB jsonencode style (minimizes diff).
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, separators=(",", ":"), ensure_ascii=False)
    print("wrote %s  (%d channel aliases, %d default vals, %d consts top-level keys)"
          % (out_json, len(cfg["channel_alias_keys"]),
             len(cfg["default_vals_keys"]), len(cfg["consts"])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
