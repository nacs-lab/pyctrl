function probe_interning()
% Engine-free ground-truth probe for SeqContext constant interning.
% Touches ONLY SeqContext/SeqVal (no SeqManager singleton, no engine, no hardware).
    here = fileparts(mfilename('fullpath'));
    addpath(fullfile(here, '..', '..', 'matlab_new', 'lib'));

    % --- +0.0 vs -0.0 : distinct or shared f64 const node? ---
    c = SeqContext();
    a = c.getValID(0.0);
    b = c.getValID(-0.0);
    fprintf('ZERO: id(0.0)=%d id(-0.0)=%d -> %s\n', a, b, pick(a==b, 'SAME', 'DISTINCT'));

    % --- NaN reuse ---
    c = SeqContext();
    a = c.getValID(NaN);
    b = c.getValID(NaN);
    fprintf('NAN: id1=%d id2=%d -> %s\n', a, b, pick(a==b, 'REUSED', 'DUPLICATED'));

    % --- +inf vs -inf distinct, and byte check ---
    c = SeqContext();
    a = c.getValID(inf);
    b = c.getValID(-inf);
    fprintf('INF: id(inf)=%d id(-inf)=%d -> %s\n', a, b, pick(a==b, 'SAME', 'DISTINCT'));
    fprintf('INF bytes: inf=%s  -inf=%s  negzero=%s\n', ...
        hexbytes(inf), hexbytes(-inf), hexbytes(-0.0));

    % --- int32 saturation on construction ---
    fprintf('INT32: int32(2^40)=%d  intmax=%d  int32(-2^40)=%d  intmin=%d\n', ...
        int32(2^40), intmax('int32'), int32(-2^40), intmin('int32'));

    % --- int8/int16/int64 all intern to the same int32 const node ---
    c = SeqContext();
    a = c.getValID(int8(23));
    b = c.getValID(int16(23));
    d = c.getValID(int64(23));
    fprintf('INTSHARE: int8=%d int16=%d int64=%d -> %s\n', a, b, d, ...
        pick(a==b && b==d, 'SHARED', 'SPLIT'));
end

function s = pick(cond, t, f)
    if cond, s = t; else, s = f; end
end

function s = hexbytes(x)
    raw = typecast(double(x), 'uint8');
    s = sprintf('%02x', raw);
end
