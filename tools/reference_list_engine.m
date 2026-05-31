function specs = reference_list_engine()
% reference_list_engine  Sequences for the ENGINE-ACCEPTS check (Phase 0).
%
% Same shape as reference_list(), but every channel name is a REAL backend name
% that matches matlab_new/config.yml (FPGA1 zynq, NiDAQ, AWG1). These are the
% sequences captured into pyctrl/tests/reference_engine/ and fed to the libnacs
% engine via Manager.create_sequence (compile-only, no init_run/start) by
% tests/test_engine_loads.py -- proving the engine accepts an externally
% produced byte array.
%
% Backend channel-name conventions (from matlab_new/expConfig.m + config.yml):
%   FPGA1/TTL<n>          TTL line on the ZYNQ FPGA (0..max_ttl_chn=55)
%   FPGA1/DDS<n>/FREQ     DDS frequency (Hz)
%   FPGA1/DDS<n>/AMP      DDS amplitude (0..1)
%   NiDAQ/Dev1/<id>       NI DAQ analog output line <id> (volts). The device
%                         channel id is REQUIRED -- 'NiDAQ/Dev1' alone is
%                         rejected ("No device channel ID in Ni DAQ channel
%                         name"). Cf. expConfig.m: 'VMOTCoil' -> 'Dev1/0'.
%
% Capture with:
%   capture_matlab_reference(fullfile('..','tests','reference_engine'), ...
%                            @reference_list_engine)

    specs = struct('name', {}, 'params', {}, 'build', {});

    specs(end+1) = entry('eng_single_ttl',   struct('len', 1e-3), @build_eng_single_ttl);
    specs(end+1) = entry('eng_ttl_pulse',    struct('len', 1e-3), @build_eng_ttl_pulse);
    specs(end+1) = entry('eng_dds_set',      struct('len', 1e-3), @build_eng_dds_set);
    specs(end+1) = entry('eng_dds_ramp',     struct('len', 2e-3), @build_eng_dds_ramp);
    specs(end+1) = entry('eng_analog_ramp',  struct('len', 2e-3), @build_eng_analog_ramp);
    specs(end+1) = entry('eng_multi',        struct('len', 1e-3), @build_eng_multi);
end

function e = entry(name, params, build)
    e = struct('name', name, 'params', params, 'build', build);
end

function s = build_eng_single_ttl()
    s = ExpSeq();
    s.addStep(1e-3).add('FPGA1/TTL1', true);
end

function s = build_eng_ttl_pulse()
    % Turn a TTL on for 1 ms then back off (two timed outputs on one channel).
    s = ExpSeq();
    s.addStep(1e-3).add('FPGA1/TTL1', true);
    s.addStep(1e-3).add('FPGA1/TTL1', false);
end

function s = build_eng_dds_set()
    s = ExpSeq();
    s.addStep(1e-3) ...
     .add('FPGA1/DDS0/FREQ', 100e6) ...
     .add('FPGA1/DDS0/AMP', 0.5);
end

function s = build_eng_dds_ramp()
    % DDS frequency ramp via a function handle (exercises the data/interp table).
    s = ExpSeq();
    s.addStep(2e-3) ...
     .add('FPGA1/DDS0/FREQ', @(t) 100e6 + t * 1e9) ...
     .add('FPGA1/DDS0/AMP', 0.5);
end

function s = build_eng_analog_ramp()
    s = ExpSeq();
    s.addStep(2e-3).add('NiDAQ/Dev1/0', @(t) t * 0.5);   % volts
end

function s = build_eng_multi()
    s = ExpSeq();
    s.addStep(1e-3) ...
     .add('FPGA1/TTL1', true) ...
     .add('FPGA1/DDS1/FREQ', 80e6) ...
     .add('FPGA1/DDS1/AMP', 0.3) ...
     .add('NiDAQ/Dev1/0', 1.0);
end
