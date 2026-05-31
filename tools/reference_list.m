function specs = reference_list()
% reference_list  Registry of representative sequences to capture (byte round-trip).
%
% Each entry has:
%   .name   file stem for the .bin / .params.json output
%   .params struct recorded alongside the bytes (for traceability)
%   .build  function handle returning a serializable ExpSeq
%
% These builders use 'Device/CH' PLACEHOLDER channel names, which serialize
% fine without the engine. They exercise the byte round-trip path (Phase 0/1):
% serialize() -> .bin -> compare_bytes.decode/encode. They are NOT fed to the
% engine (the real engine config does not know these device names). For the
% engine-accepts check use reference_list_engine.m (real config.yml names).
%
% Builders mirror the patterns proven in matlab_new/lib/test/TestExpSeq.m, and
% span the ~12 representative shapes called for in PYTHON_FRONTEND_PLAN.m Phase 0:
% empty, single TTL, DDS freq/amp set, analog ramp, two-channel, multi-channel,
% nested subsequences, branch/conditional, globals, measurement, floating/align,
% and a long sequence.

    specs = struct('name', {}, 'params', {}, 'build', {});

    specs(end+1) = entry('empty',          struct(),                 @() ExpSeq());
    specs(end+1) = entry('single_ttl',     struct('len', 1e-3),      @build_single_ttl);
    specs(end+1) = entry('dds_set',        struct(),                 @build_dds_set);
    specs(end+1) = entry('analog_ramp',    struct('len', 2e-3),      @build_analog_ramp);
    specs(end+1) = entry('two_channel',    struct('len', 1e-3),      @build_two_channel);
    specs(end+1) = entry('multi_channel',  struct('len', 1e-3),      @build_multi_channel);
    specs(end+1) = entry('nested_subseq',  struct('len', 0.4),       @build_nested_subseq);
    specs(end+1) = entry('conditional',    struct(),                 @build_conditional);
    specs(end+1) = entry('with_global',    struct(),                 @build_with_global);
    specs(end+1) = entry('with_measure',   struct(),                 @build_with_measure);
    specs(end+1) = entry('floating_align', struct('len', 5),         @build_floating_align);
    specs(end+1) = entry('long_seq',       struct('nsteps', 64),     @build_long_seq);
end

function e = entry(name, params, build)
    e = struct('name', name, 'params', params, 'build', build);
end

function s = build_single_ttl()
    s = ExpSeq();
    s.addStep(1e-3).add('Device1/CH1', true);
end

function s = build_dds_set()
    % DDS frequency + amplitude set (constant values on two sub-channels).
    s = ExpSeq();
    s.addStep(1e-3) ...
     .add('Device1/FREQ', 100e6) ...
     .add('Device1/AMP', 0.5);
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

function s = build_multi_channel()
    s = ExpSeq();
    s.addStep(1e-3) ...
     .add('Device1/CH1', true) ...
     .add('Device2/CH3', -1) ...
     .add('Device3/CH7', 2.5) ...
     .add('Device1/CH2', @(t) t * 100);
end

function s = build_nested_subseq()
    % Nested subsequence with a measurement and a function-handle ramp,
    % plus a background subsequence (mirrors TestExpSeq.test2 without globals).
    s = ExpSeq();
    s.addStep(@subseq, 0.4);
    s.addBackground(@subseq, 0.4);
    s.waitAll();

    function subseq(s, len)
        m = s.addMeasure('Device2/CH2');
        s.addStep(len) ...
         .add('Device2/CH2', 3.4) ...
         .add('Device3/CH1', @(t) t * 5 - m);
        s.add('Device2/CH2', 0) ...
         .add('Device3/CH1', 0);
    end
end

function s = build_conditional()
    % Conditional (branch) steps: a false branch (skipped) and a true branch
    % (mirrors TestExpSeq.test1's conditional() usage).
    s = ExpSeq();
    s.addStep(1).add('Device1/CH1', 4);
    s.conditional(false).addStep(0.1).add('Device1/CH5', 3);
    s.conditional(true).addStep(0.1004).add('Device2/CH3', -1);
end

function s = build_with_global()
    s = ExpSeq();
    g = s.newGlobal();
    s.add('Device2/CH2', g + 2);
    s.addStep(1e-3).add('Device2/CH2', 0);
end

function s = build_with_measure()
    % A measurement feeding a conditional wait (exercises the measures table).
    s = ExpSeq();
    s.addStep(1).add('Device1/CH1', 4);
    m1 = s.addMeasure('Device2/CH5');
    s.conditional(m1 < 0).wait(3.4);
    s.addStep(1.2).add('Device1/CH5', @(t) t - 2.3);
end

function s = build_floating_align()
    % Floating step pinned by setEndTime then waited for (repeatable serialize;
    % mirrors TestExpSeq.test6, which is byte-stable across repeated serialize()).
    s = ExpSeq();
    g = s.newGlobal();
    step = s.addFloating(5);
    s.wait(g * 4);
    step.setEndTime(endTime(s));
    s.waitFor(step);
end

function s = build_long_seq()
    % A long single-channel pulse train (exercises a large times/outputs table).
    % Step length must be >= 1 tick at the capture rate (override_tick_per_sec
    % = 1000 => 1e-3 s == 1 tick); a shorter step rounds to 0 and is rejected.
    s = ExpSeq();
    for i = 1:64
        s.addStep(1e-3).add('Device1/CH1', mod(i, 2) == 0);
    end
end
