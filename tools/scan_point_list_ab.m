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
    specs(end+1) = entry('bluelac',         @build_bluelac,         'BlueTweezerLoadingSeq');
    specs(end+1) = entry('spectrum556',     @build_spectrum556,     'PushoutSurvivalSeq');
    specs(end+1) = entry('imaginglifetime', @build_imaginglifetime, 'ImagingPushoutSurvivalSeq');
    specs(end+1) = entry('coolingx2d',      @build_coolingx2d,      'ImagingPushoutSurvivalSeq');
    specs(end+1) = entry('coolingrnr',      @build_coolingrnr,      'ReleaseRecaptureSeq');
    specs(end+1) = entry('releasetime',     @build_releasetime,     'ReleaseRecaptureSeq');
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

function g = build_imaginglifetime()
    % ImagingLifetimeScan.m / TimeImagingScan: IMAGING during the push-out -- the push-out
    % beams run at the Imag399/Cool556 imaging amplitudes (not 0). ImagingPushoutSurvivalSeq ->
    % PushouthXStep reads Pushout.Time + Blue.{Freq,Amp} + Green.{X,h}.{Freq,Amp}; freqs + amps
    % derived from Consts() exactly as the .m.
    g = ScanGroup();
    g().Pushout.Blue.Freq    = Consts().Resonance399Freq + Consts().Imag399.FreqDetuning;
    g().Pushout.Blue.Amp     = Consts().Imag399.Amp;
    g().Pushout.Green.X.Freq = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.X.FreqDetuning;
    g().Pushout.Green.X.Amp  = Consts().Imag399.Cool556.X.Amp;
    g().Pushout.Green.h.Freq = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.h.FreqDetuning;
    g().Pushout.Green.h.Amp  = Consts().Imag399.Cool556.h.Amp;
    g().Pushout.Time.scan(1) = [0.005, 0.1, 1, 2, 4, 8];   % 6 explicit points (s)
end

function g = build_coolingx2d()
    % CoolingScan.py x2d default: 2-D Pushout.Green.X.{Freq,Amp}, Blue.Amp=0.3, h fixed.
    % Freq = Resonance556mj0 + (0.10:0.02:0.26)*1e6 (dim 1); Amp = 0.1:0.02:0.28 (dim 2). 9x10=90 pts.
    g = ScanGroup();
    g().Pushout.Time = 100e-3;
    g().Pushout.Blue.Freq = Consts().Resonance399Freq + Consts().Imag399.FreqDetuning;
    g().Pushout.Blue.Amp = 0.3;
    g().Pushout.Green.h.Freq = Consts().Resonance556mj0Freq + Consts().Imag399.Cool556.h.FreqDetuning;
    g().Pushout.Green.h.Amp = Consts().Imag399.Cool556.h.Amp;
    g().Pushout.Green.X.Freq.scan(1) = Consts().Resonance556mj0Freq + (0.10:0.02:0.26)*1e6;
    g().Pushout.Green.X.Amp.scan(2) = 0.1:0.02:0.28;
end

function g = build_coolingrnr()
    % CoolingScan_RNR.m FreqCooling556Scan: release-and-recapture 556 h-beam scan.
    % ReleaseRecaptureSeq -> Cool556hXStep reads Cool556.{Time, X.FreqDetuning, X.Amp,
    % h.FreqDetuning, h.Amp}; ReleaseRecaptureStep reads ReleaseRecapture.Time. The swept h
    % detuning is the BARE detuning (Hz) -- the step adds Resonance556mj0Freq. 9x11 = 99 pts.
    g = ScanGroup();
    g().Cool556.Time = 5e-3;
    g().Cool556.X.FreqDetuning = 0.11e6;
    g().Cool556.X.Amp = 0.16;
    g().Cool556.h.FreqDetuning.scan(1) = (0.08:0.01:0.16)*1e6;   % 9 points
    g().Cool556.h.Amp.scan(2) = 0:0.02:0.2;                      % 11 points
    g().ReleaseRecapture.Time = 25e-6;
end

function g = build_releasetime()
    % ReleaseRecaptureScan.m ReleaseTimeScan: release-and-recapture release-TIME sweep
    % (atom-temperature measurement). ReleaseRecaptureSeq -> ReleaseRecaptureStep reads
    % ReleaseRecapture.Time (-> s.wait free-flight gap); Imag399Step reads Imag399.ExposureTime
    % (s.wait), SLMStep reads SLM.VServo (s.add 'VSLMservo'), Cool556hXStep reads Cool556.Time
    % (s.wait). ReleaseRecapture.Hold is set but UNREAD by the step (no byte effect; mirrored
    % only for faithfulness). The colon (0:1:50) is INTEGER-valued, so the *1e-6 is an exact
    % per-element scalar multiply -- no 1-ULP colon trap (no matlab_colon needed Python-side).
    g = ScanGroup();
    g().Imag399.ExposureTime = 100e-3;
    g().SLM.VServo = 5;
    g().Cool556.Time = 5e-3;
    g().ReleaseRecapture.Hold = 0;
    g().ReleaseRecapture.Time.scan(1) = (0:1:50)*1e-6;   % 51 points
end
