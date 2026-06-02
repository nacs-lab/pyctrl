function specs = scan_point_list()
% scan_point_list  Real (scan, sequence) pairs for the Phase-4 W6 per-point byte oracle.
%
% Each entry pairs a ScanGroup builder with the sequence function the real YbScans file
% runs it through, so the capture can do exactly what RunScans/runSeq2 does per shot:
%   params = getseq(g, n);  s = ExpSeq(params);  seqfn(s);  serialize(s)
%
% The builders replicate REAL matlab_new/YbScans/* parameter blocks (trimmed scan lengths;
% in-body hardware calls like OrcaInit and the dbstack scanname/scanfilename metadata are
% dropped -- neither affects the serialized bytes). Each builder MUST stay byte-for-byte
% equivalent to its twin in pyctrl/tests/test_scan_point_oracle.py.
%
% Both paired sequences are in the Phase-3 byte-verified ok corpus, so their no-param build
% already matches MATLAB; W6 adds the scan dimension (ExpSeq(getseq(n)) merges the per-point
% params onto consts) and proves each point still matches.

    specs = struct('name', {}, 'build', {}, 'seq', {});
    specs(end+1) = entry('spectrum399',  @build_spectrum399,  'PushoutSurvival399Seq');
    specs(end+1) = entry('imaging_hist', @build_imaging_hist, 'PushoutSurvivalSeq');
end

function e = entry(name, build, seq)
    e = struct('name', name, 'build', build, 'seq', seq);
end

function g = build_spectrum399()
    % FreqPushOut399Scan (Spectrum399Scan.m): a 1-D scan of Pushout.Blue.Freq.
    g = ScanGroup();
    g().Pushout.Blue.Amp = 0.25;
    g().Pushout.Blue.Freq.scan(1) = (220:35:360) * 1e6;   % 5 points
    g().Pushout.Time = 10e-3;
    g.runp().NumPerGroup = 10000;
    g.runp().NumImages = 2;
    g.runp().Scramble = 1;
end

function g = build_imaging_hist()
    % imagingScan (ImagingHistScan.m): a 2-D grid over Imag399.FreqDetuning x Imag399.Amp.
    g = ScanGroup();
    g().Imag399.ExposureTime = 100e-3;
    g().SLM.VServo = 1;
    g().Imag399.FreqDetuning.scan(1) = [-5, 0] * 1e6;
    g().Imag399.Amp.scan(2) = [0.2, 0.3];
    g().Pushout.Green.Amp = 0;
    g().Pushout.Blue.Amp = 0;
    g().Pushout.Time = 10e-3;
    g.runp().NumImages = 2;
    g.runp().Scramble = 1;
end
