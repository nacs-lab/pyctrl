function specs = scan_point_list_ab()
% scan_point_list_ab  (scan, sequence) pairs for the BlueLAC + Spectrum556 A/B byte oracle.
%
% Twins of pyctrl/YbScans/{BlueLACScan,Spectrum556Scan}.py build(). Feed this to
% capture_scan_point_reference (which does exactly RunScans/runSeq2 per shot --
% params = getseq(g,n); s = ExpSeq(params); seqfn(s); serialize(s)) to capture per-point bytes:
%   matlab -batch "cd <pyctrl/tools>; capture_scan_point_reference( ...
%       fullfile(pwd,'..','tests','reference_scan_point','ab_reference.json'), @scan_point_list_ab)"
%
% Only the byte-affecting params are set (the dbstack scanname/scanfilename, debug=0, and runp
% are dropped -- none enters the serialized bytes). The Python twin MUST stay byte-for-byte
% equivalent: it generates the colon sweeps with scan_export.matlab_colon (a bit-identical
% reproduction of MATLAB's colon operator -- a naive a+k*step differs by 1 ULP).

    specs = struct('name', {}, 'build', {}, 'seq', {});
    specs(end+1) = entry('bluelac',     @build_bluelac,     'BlueTweezerLoadingSeq');
    specs(end+1) = entry('spectrum556', @build_spectrum556, 'PushoutSurvivalSeq');
end

function e = entry(name, build, seq)
    e = struct('name', name, 'build', build, 'seq', seq);
end

function g = build_bluelac()
    % BlueLACScan.m: the documented scannedFreq = -(2:0.6:9)*1e6 wired as the active sweep.
    % BlueTweezerLoadingSeq -> BlueLACStep reads LAC.BlueLAC.FreqDetuning (-> Freq556RydbergMOTh).
    g = ScanGroup();
    g().LAC.BlueLAC.FreqDetuning.scan(1) = -(2:0.6:9)*1e6;    % 12 points
end

function g = build_spectrum556()
    % Spectrum556Scan.m: the active "|mj|=1, check trap depth" block.
    % PushoutSurvivalSeq -> PushoutStep reads Pushout.Green.Freq/Amp + Pushout.Time.
    g = ScanGroup();
    g().Pushout.Green.Amp = 0.18;
    g().Pushout.Time = 20e-3;
    g().Pushout.Green.Freq.scan(1) = (103.5:0.1:106.5)*1e6;  % 31 points
end
