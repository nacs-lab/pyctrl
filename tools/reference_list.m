function specs = reference_list()
% reference_list  Registry of representative sequences to capture.
%
% Each entry has:
%   .name   file stem for the .bin / .params.json output
%   .params struct recorded alongside the bytes (for traceability)
%   .build  function handle returning a serializable ExpSeq
%
% These builders use 'Device/CH' placeholder channel names, which serialize
% fine without the engine. They exercise the byte round-trip path (Phase 0/1).
% For the engine-accepts check (pytest -m needs_engine), add entries whose
% channel names match config.yml and capture them into tests/reference_engine/.
%
% Builders mirror the patterns proven in matlab_new/lib/test/TestExpSeq.m.

    specs = struct('name', {}, 'params', {}, 'build', {});

    specs(end+1) = entry('empty',       struct(),                @() ExpSeq());
    specs(end+1) = entry('single_ttl',  struct('len', 1e-3),     @build_single_ttl);
    specs(end+1) = entry('analog_ramp', struct('len', 2e-3),     @build_analog_ramp);
    specs(end+1) = entry('two_channel', struct('len', 1e-3),     @build_two_channel);
    specs(end+1) = entry('with_global', struct(),                @build_with_global);
end

function e = entry(name, params, build)
    e = struct('name', name, 'params', params, 'build', build);
end

function s = build_single_ttl()
    s = ExpSeq();
    s.addStep(1e-3).add('Device1/CH1', true);
end

function s = build_analog_ramp()
    s = ExpSeq();
    s.addStep(2e-3).add('Device1/CH2', @(t) t * 500);   % function-handle ramp
end

function s = build_two_channel()
    s = ExpSeq();
    s.addStep(1e-3) ...
     .add('Device1/CH1', true) ...
     .add('Device2/CH3', -1);
end

function s = build_with_global()
    s = ExpSeq();
    g = s.newGlobal();
    s.add('Device2/CH2', g + 2);
    s.addStep(1e-3).add('Device2/CH2', 0);
end
