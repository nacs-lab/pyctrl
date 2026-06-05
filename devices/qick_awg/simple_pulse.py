"""simple_pulse.py -- QICK pulse + program JSON builders.

Pure ports of the matlab_new QICK helpers, returning JSON-serializable dicts (the upload itself is
the client's job -- see :mod:`fpga_awg_client` / :mod:`fpga_awg_manager`):

  * :func:`simple_pulse_cfg`  == ``uploadSimplePulse.m`` (minus the file write + upload).
  * :func:`compile_chn`       == ``compileCHN.m`` (channels -> ``{ch0, ch1}`` prog_structure).
  * :func:`simple_prog_cfg`   == ``uploadSimpleProg.m`` (minus the file write + upload).

A program channel is a list of **tokens**: either a pulse-name ``str`` or a :class:`Loop` (use
:func:`loop`). Keeping loops STRUCTURED (rather than the MATLAB ``'loop(10,[Wait])'`` string) lets
:func:`compile_chn` apply a ``name_map`` to pulse-name leaves recursively -- which is how
:class:`~fpga_awg_manager.FPGAAWGManager` namespaces pulses so multiple programs can coexist on the
server for per-shot picking.
"""
from collections import namedtuple

# A bounded repeat of a token body: loop(count, [body...]) -> "loop(<count>,[<body>])".
Loop = namedtuple("Loop", ["count", "body"])


def loop(count, body):
    """Build a :class:`Loop` token. ``body`` is a token list (or a single token)."""
    if isinstance(body, (str, Loop)):
        body = [body]
    return Loop(int(count), list(body))


def simple_pulse_cfg(name, freq, gain, length, phase=0.0, style="const", mode="oneshot"):
    """A const-style pulse cfg dict (== ``uploadSimplePulse.m``'s struct).

    Args:
        name: pulse name (server-side key).
        freq: carrier frequency (MHz, as the MATLAB passed it).
        gain: amplitude / DAC gain.
        length: pulse length in **ns** (the MATLAB ``dur``).
        phase: phase in **degrees** (the MATLAB converts rad->deg before calling).
    """
    return {
        "name": name,
        "style": style,
        "freq": freq,
        "gain": gain,
        "phase": phase,
        "length": length,
        "mode": mode,
    }


def render_tokens(tokens, name_map=None):
    """Render a token list to the ``[a,loop(10,[b]),c]`` string (== one ``compileCHN`` channel).

    ``name_map`` (optional) substitutes pulse-name string tokens (unknown tokens pass through), so a
    program's pulses can be namespaced without string surgery on already-built loop strings.
    """
    name_map = name_map or {}
    parts = []
    for tok in tokens:
        if isinstance(tok, Loop):
            parts.append("loop(%s,%s)" % (tok.count, render_tokens(tok.body, name_map)))
        elif isinstance(tok, str):
            parts.append(name_map.get(tok, tok))
        else:                                       # numeric token (MATLAB num2str fallback)
            parts.append(str(tok))
    return "[" + ",".join(parts) + "]"


def compile_chn(channels, name_map=None):
    """Compile 1 or 2 channels of tokens into a ``{ch0, ch1}`` prog_structure (== ``compileCHN.m``)."""
    if len(channels) == 0 or len(channels) > 2:
        raise ValueError("compile_chn supports 1 or 2 channels, got %d" % len(channels))
    out = {"ch0": render_tokens(channels[0], name_map)}
    if len(channels) == 2:
        out["ch1"] = render_tokens(channels[1], name_map)
    return out


def simple_prog_cfg(name, prog_structure):
    """A program cfg dict (== ``uploadSimpleProg.m``'s struct)."""
    return {"name": name, "prog_structure": prog_structure}
