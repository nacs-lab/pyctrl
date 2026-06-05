function capture_branch_callback_probe(out_json)
% capture_branch_callback_probe  Serialize the minimal branch+callback probe.
%
% Builds the s1/s2/s3 + condBranch + regBeforeStart/regBeforeBSeq/regAfterBSeq/
% regAfterBranch/regAfterEnd skeleton that SLMRearrangement (RearrangeCommSeq /
% RearrangeCommSeq2) relies on, ENGINE-FREE, and writes its serialize() bytes
% (hex) to OUT_JSON so the pyctrl byte test (tests/test_branch_callback_probe.py)
% can assert pyctrl's serialize() == MATLAB's for the exact same probe.
%
% SAFETY -- this only BUILDS + serialize()s the tree:
%   * serialize() never runs the deferred regBefore*/regAfter* callbacks
%     (they are no-ops here anyway), so building touches no hardware.
%   * SeqManager.override_tick_per_sec(1e12) keeps the libnacs engine unloaded.
%   * The body uses only a synthetic channel ('Device1/CH1') + condBranch +
%     newBasicSeq -- no AWG / SLM / camera / FPGA code path is reached.
%   * Run it as a SEPARATE headless process (matlab -batch); it adds paths and
%     touches the SeqManager singleton only within its own process.
% Config = the REAL matlab_new/expConfig.m (the same config pyctrl loads via
% SeqConfig.load_real() -> expConfig.py), so the bytes are directly comparable.
%
%   capture_branch_callback_probe                 % -> tests/reference_branch_probe/
%   capture_branch_callback_probe(out_json)       % explicit path

    here = fileparts(mfilename('fullpath'));
    repo = fullfile(here, '..', '..');
    mn   = fullfile(repo, 'matlab_new');
    if nargin < 1 || isempty(out_json)
        out_json = fullfile(here, '..', 'tests', 'reference_branch_probe', ...
                            'probe_reference.json');
    end
    odir = fileparts(out_json);
    if ~exist(odir, 'dir'), mkdir(odir); end

    % Real framework + real expConfig.m on the path (mirrors
    % capture_ybseqs_reference.m: NOT +archived / .claude worktrees).
    for d = {mn, fullfile(mn, 'lib'), fullfile(mn, 'YbSteps'), ...
             fullfile(mn, 'YbSeqs'), fullfile(mn, 'YbScans'), ...
             fullfile(mn, 'YbScans', 'scanConfig'), ...
             fullfile(mn, 'YbRearrangement'), fullfile(mn, 'YbExptCtrl')}
        if exist(d{1}, 'dir'), addpath(d{1}); end
    end

    % Production tick rate (config.yml: 1e12 == 1 ps), engine stays unloaded.
    SeqManager.override_tick_per_sec(1e12);
    cleanup = onCleanup(@() SeqManager.override_tick_per_sec(0)); %#ok<NASGU>

    try
        s = ExpSeq();
        build_probe(s);
        b = serialize(s);
        hexb = lower(sprintf('%02x', typecast(int8(b), 'uint8')));
        status = 'ok';
        nbytes = numel(b);
    catch e
        msg = regexprep(e.message, '[\"\\\r\n\t]', ' ');
        if numel(msg) > 160, msg = msg(1:160); end
        hexb = '';
        status = ['err:' msg];
        nbytes = 0;
    end

    out = sprintf('[{"name":"BranchCallbackProbe","status":"%s","bytes":"%s"}]', ...
                  status, hexb);
    fid = fopen(out_json, 'w');
    fwrite(fid, out, 'char');
    fclose(fid);
    fprintf('wrote BranchCallbackProbe (status=%s, %d bytes) to %s\n', ...
            status, nbytes, out_json);
end


function build_probe(s)
% Minimal 3-bseq branch+callback skeleton -- mirror of build_probe() in
% tests/test_branch_callback_probe.py. Root -> s2 -> s3 via newBasicSeq +
% condBranch(true, ...); one trivial synthetic TTL step per bseq; the full
% per-bseq callback triple on every bseq plus regBeforeStart / regAfterEnd on
% the root. Callbacks are no-ops (zero byte impact); pyctrl exercises their
% firing separately.
    s.regBeforeStart(@noop);
    s.regBeforeBSeq(@noop);
    s.regAfterBSeq(@noop);
    s.regAfterBranch(@noop);
    s.addStep(1).add('Device1/CH1', 1);

    s2 = s.newBasicSeq();
    s.condBranch(true, s2);
    s2.regBeforeBSeq(@noop);
    s2.regAfterBSeq(@noop);
    s2.regAfterBranch(@noop);
    s2.addStep(1).add('Device1/CH1', 0);

    s3 = s.newBasicSeq();
    s2.condBranch(true, s3);
    s3.regBeforeBSeq(@noop);
    s3.regAfterBSeq(@noop);
    s3.regAfterBranch(@noop);
    s3.addStep(1).add('Device1/CH1', 1);

    s.regAfterEnd(@noop);
end


function noop(~)
end
