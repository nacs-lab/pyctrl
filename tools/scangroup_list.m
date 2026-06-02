function specs = scangroup_list()
% scangroup_list  Registry of ScanGroups to capture for the W4 expansion oracle.
%
% Each entry has:
%   .name   key under which the capture is stored in scangroup_reference.json
%   .build  function handle returning a fully-built ScanGroup
%
% IMPORTANT: every builder here MUST be kept byte-for-byte equivalent to its twin
% in pyctrl/tests/test_scan_group_oracle.py (the Python BATTERY). The two builders
% use different syntax for a scan axis -- MATLAB `.scan(dim) = vals`, Python
% `.scan(dim, vals)` -- but must produce the same group. The oracle's whole job is
% to prove the Python column-major getseq() expansion matches MATLAB's, so the
% INPUTS must match by construction.
%
% Production-shaped tier (single-group: fixed, 1-D, 2-D grid, nested, float) plus
% one base-merge case (g(n) override + nested) to cross-check getfullscan.

    specs = struct('name', {}, 'build', {});
    specs(end+1) = entry('fixed_only',         @build_fixed_only);
    specs(end+1) = entry('scan_1d',            @build_scan_1d);
    specs(end+1) = entry('scan_2d',            @build_scan_2d);
    specs(end+1) = entry('awg_like',           @build_awg_like);
    specs(end+1) = entry('mixed_float_1d',     @build_mixed_float_1d);
    specs(end+1) = entry('two_scan_basemerge', @build_two_scan_basemerge);
end

function e = entry(name, build)
    e = struct('name', name, 'build', build);
end

function g = build_fixed_only()
    g = ScanGroup();
    g().a = 1;
    g().b.c = 2;
    g().s = 'hello';
end

function g = build_scan_1d()
    g = ScanGroup();
    g().amp = 0.5;
    g().freq.scan(1) = [10, 20, 30, 40];
end

function g = build_scan_2d()
    g = ScanGroup();
    g().fixed = 7;
    g().c.scan(1) = [1, 2, 3];
    g().d.scan(2) = [10, 20];
    g.runp().NumImages = 2;
    g.runp().NumPerGroup = 16;
    g.runp().Scramble = 1;
end

function g = build_awg_like()
    g = ScanGroup();
    g().AWG.AWG556.pulse_width_us.scan(1) = [1, 2, 3, 4];
    g().AWG.AWG556.carrier_freq_MHz.scan(2) = [100, 110];
    g().Pushout.delay = 1.3e-6;
end

function g = build_mixed_float_1d()
    g = ScanGroup();
    g().t.scan(1) = [0.1, 0.2, 0.3, 0.4, 0.5];
    g().n = 16;
end

function g = build_two_scan_basemerge()
    g = ScanGroup();
    g().a = 1;
    g().b = 2;
    g().c.scan(1) = [1, 2, 3];
    g(1).c = 3;                 % scan 1 fixes c -> shadows the base scan axis
    g().d.scan(2) = [1, 2];
    g(2).d = 0;                 % scan 2 fixes d -> shadows the base scan axis
    g(2).k.a.b.c = 2;           % nested fixed param on scan 2
end
