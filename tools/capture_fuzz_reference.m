function capture_fuzz_reference(programs_json, out_json)
% capture_fuzz_reference  MATLAB ground truth for the SeqContext differential fuzzer.
%
% Replays the language-neutral program specs in PROGRAMS_JSON (produced by
% tools/fuzz_programs.py) with the real SeqVal/SeqContext and writes the
% node/data/global serialized tables (as hex) to OUT_JSON. The Python side
% (tests/test_differential_fuzz.py) rebuilds the same specs and asserts equality.
%
% Engine-free and hardware-free: touches ONLY SeqContext/SeqVal -- no SeqManager
% singleton, no serialize() of a full sequence, no engine load. Safe any time, but
% run it in a SEPARATE MATLAB session from a live experiment (it adds lib to the
% path and uses the base workspace only transiently inside this function).
%
%   capture_fuzz_reference                      % uses the default reference_fuzz paths
%   capture_fuzz_reference(progs, out)          % explicit paths

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(programs_json)
        programs_json = fullfile(here, '..', 'tests', 'reference_fuzz', 'programs.json');
    end
    if nargin < 2 || isempty(out_json)
        out_json = fullfile(here, '..', 'tests', 'reference_fuzz', 'fuzz_reference.json');
    end
    addpath(fullfile(here, '..', '..', 'matlab_new', 'lib'));

    txt = fileread(programs_json);
    programs = jsondecode(txt);

    n = numel(programs);
    parts = cell(1, n);
    for i = 1:n
        prog = getel(programs, i);
        [node_hex, data_hex, global_hex] = build_one(prog);
        parts{i} = sprintf('{"node":"%s","data":"%s","global":"%s"}', ...
                           node_hex, data_hex, global_hex);
    end
    out = ['[', strjoin(parts, ','), ']'];

    fid = fopen(out_json, 'w');
    fwrite(fid, out, 'char');
    fclose(fid);
    fprintf('wrote %d program references to %s\n', n, out_json);
end


function [node_hex, data_hex, global_hex] = build_one(prog)
    ctx = SeqContext();

    % --- create globals / measures up front (fixed order) ---
    gtypes = prog.globals;
    globals = cell(1, numel(gtypes));
    for k = 1:numel(gtypes)
        [g, ~] = newGlobal(ctx, double(gtypes(k)));
        globals{k} = g;
    end
    measures = cell(1, prog.nmeasure);
    for k = 1:prog.nmeasure
        [m, ~] = newMeasure(ctx);
        measures{k} = m;
    end

    env = struct('ctx', ctx, 'globals', {globals}, 'measures', {measures}, 'values', {{}});

    % --- replay steps ---
    steps = prog.steps;
    for j = 1:numel(steps)
        step = getel(steps, j);
        val = apply_step(env, step);
        env.values{end + 1} = val; %#ok<AGROW>
    end

    % --- emits (getValID) in recorded order ---
    emits = prog.emits;
    for j = 1:numel(emits)
        ref = getel(emits, j);
        getValID(ctx, resolve(env, ref));
    end

    node_hex = tohex(nodeSerialized(ctx));
    data_hex = tohex(dataSerialized(ctx));
    global_hex = tohex(globalSerialized(ctx));
end


function val = apply_step(env, step)
    op = char(step.op);
    a = resolve(env, step.a);
    switch op
        % --- unary ---
        case 'abs',    val = abs(a);
        case 'exp',    val = exp(a);
        case 'floor',  val = floor(a);
        case 'log',    val = log(a);
        case 'sqrt',   val = sqrt(a);
        case 'sin',    val = sin(a);
        case 'cos',    val = cos(a);
        case 'atan',   val = atan(a);
        case 'erf',    val = erf(a);
        case 'round',  val = round(a);
        case 'not',    val = ~a;
        case 'neg',    val = -a;
        case 'uplus',  val = +a;
        % --- binary ---
        case 'add',    val = a + resolve(env, step.b);
        case 'sub',    val = a - resolve(env, step.b);
        case 'mul',    val = a * resolve(env, step.b);
        case 'div',    val = a / resolve(env, step.b);
        case 'pow',    val = a ^ resolve(env, step.b);
        case 'ldiv',   val = ldivide(a, resolve(env, step.b));
        case 'atan2',  val = atan2(a, resolve(env, step.b));
        case 'hypot',  val = hypot(a, resolve(env, step.b));
        case 'rem',    val = rem(a, resolve(env, step.b));
        case 'max',    val = max(a, resolve(env, step.b));
        case 'min',    val = min(a, resolve(env, step.b));
        case 'and',    val = a & resolve(env, step.b);
        case 'or',     val = a | resolve(env, step.b);
        case 'xor',    val = xor(a, resolve(env, step.b));
        case 'lt',     val = a < resolve(env, step.b);
        case 'gt',     val = a > resolve(env, step.b);
        case 'le',     val = a <= resolve(env, step.b);
        case 'ge',     val = a >= resolve(env, step.b);
        case 'eq',     val = a == resolve(env, step.b);
        case 'ne',     val = a ~= resolve(env, step.b);
        % --- ternary ---
        case 'interp', val = interpolate(a, resolve(env, step.b), ...
                                         resolve(env, step.c), reshape(double(step.data), 1, []));
        case 'ifelse', val = ifelse(a, resolve(env, step.b), resolve(env, step.c));
        otherwise, error('unknown op %s', op);
    end
end


function v = resolve(env, ref)
    k = char(ref.k);
    switch k
        case 'g', v = env.globals{ref.idx + 1};
        case 'm', v = env.measures{ref.idx + 1};
        case 'a', v = getArg(env.ctx, ref.idx);
        case 'v', v = env.values{ref.idx + 1};
        case 'f', v = double(ref.num);
        case 'i', v = int32(ref.num);
        case 'b', v = logical(ref.num);
        otherwise, error('bad ref kind %s', k);
    end
end


function el = getel(arr, i)
    % jsondecode yields a struct array for uniform objects, a cell array otherwise.
    if iscell(arr)
        el = arr{i};
    else
        el = arr(i);
    end
end


function s = tohex(int8row)
    if isempty(int8row)
        s = '';
        return;
    end
    s = sprintf('%02x', typecast(int8(int8row), 'uint8'));
end
