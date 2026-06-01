function capture_ybseqs_reference(seqs_dir, out_json)
% capture_ybseqs_reference  Serialize real experiment sequences for the reader test.
%
% Walks matlab_new/YbSeqs, builds each sequence ENGINE-FREE, and writes its
% serialize() bytes (hex) to OUT_JSON so the pyctrl reader round-trip test
% (tests/test_ybseqs_roundtrip_live.py) can verify the byte-format reader against
% real production sequences. Records a per-sequence status (ok / skipped / error).
%
% SAFETY -- this only BUILDS + serialize()s the tree:
%   * serialize() never runs the deferred regBeforeStart/regAfterEnd callbacks
%     (camera/AWG/server setup), so building does not touch that hardware.
%   * SeqManager.override_tick_per_sec(1000) keeps the libnacs engine unloaded.
%   * A name DENYLIST skips sequences that drive AWG / SLM / AOD / camera /
%     picomotor hardware directly in their body, so we never even attempt them.
%   * Run it as a SEPARATE headless process (matlab -batch); it adds paths and
%     touches the SeqManager singleton only within its own process.
% Still: prefer a maintenance window. Build is engine-free and callback-free, but
% it is real sequence code -- run it when you would be comfortable building a shot.
%
%   capture_ybseqs_reference                 % default YbSeqs -> reference_ybseqs/
%   capture_ybseqs_reference(dir, out)       % explicit paths

    here = fileparts(mfilename('fullpath'));
    repo = fullfile(here, '..', '..');
    mn = fullfile(repo, 'matlab_new');
    if nargin < 1 || isempty(seqs_dir), seqs_dir = fullfile(mn, 'YbSeqs'); end
    if nargin < 2 || isempty(out_json)
        out_json = fullfile(here, '..', 'tests', 'reference_ybseqs', 'ybseqs_reference.json');
    end
    odir = fileparts(out_json);
    if ~exist(odir, 'dir'), mkdir(odir); end

    % --- real framework on the path (NOT +archived / .claude worktrees) ---
    for d = {mn, fullfile(mn,'lib'), fullfile(mn,'YbSeqs'), fullfile(mn,'YbSteps'), ...
             fullfile(mn,'YbScans'), fullfile(mn,'YbScans','scanConfig'), ...
             fullfile(mn,'YbRearrangement'), fullfile(mn,'YbFPGAWaveforms'), ...
             fullfile(mn,'YbExptCtrl')}
        if exist(d{1}, 'dir'), addpath(d{1}); end
    end

    % Sequences that drive hardware directly in-body -> never attempt to build.
    denylist = 'AWG|SLM|AOD|Orca|Pico|Rearrange|HandOver|Trigger|SingleMove|setAWG|[Mm]odulation';

    % Real production tick rate (config.yml: 1e12 == 1 ps). The synthetic
    % reference_list.m builders could use 1000 because their steps are ms-scale;
    % real sequences use us-scale steps that round to 0 ticks at 1000/s and are
    % rejected ("Time offset/length must be positive"). Engine stays unloaded.
    SeqManager.override_tick_per_sec(1e12);
    cleanup = onCleanup(@() SeqManager.override_tick_per_sec(0)); %#ok<NASGU>

    files = dir(fullfile(seqs_dir, '*.m'));
    parts = {};
    n_ok = 0;
    for i = 1:numel(files)
        name = files(i).name(1:end-2);   % strip .m
        if ~isempty(regexp(name, denylist, 'once'))
            parts{end+1} = entry_json(name, 'skip:hardware-driver', ''); %#ok<AGROW>
            continue;
        end
        [hexb, status] = try_build(name);
        if strcmp(status, 'ok'), n_ok = n_ok + 1; end
        parts{end+1} = entry_json(name, status, hexb); %#ok<AGROW>
    end
    out = ['[', strjoin(parts, ','), ']'];

    fid = fopen(out_json, 'w');
    fwrite(fid, out, 'char');
    fclose(fid);
    fprintf('wrote %d sequences (%d serialized ok) to %s\n', numel(files), n_ok, out_json);
end


function [hexb, status] = try_build(name)
    hexb = '';
    try
        fn = str2func(name);
        nin = nargin(fn);
        if nin == 0
            s = fn();
        elseif nin == 1
            s = fn(ExpSeq());           % takes a configured ExpSeq, returns it
        else
            status = sprintf('skip:nargin=%d', nin);
            return;
        end
        if ~isa(s, 'ExpSeq')
            status = 'skip:not-an-ExpSeq';
            return;
        end
        b = serialize(s);
        hexb = lower(sprintf('%02x', typecast(int8(b), 'uint8')));
        status = 'ok';
    catch e
        msg = regexprep(e.message, '[\"\\\r\n\t]', ' ');
        if numel(msg) > 160, msg = msg(1:160); end
        status = ['err:' msg];
    end
end


function s = entry_json(name, status, hexb)
    s = sprintf('{"name":"%s","status":"%s","bytes":"%s"}', name, status, hexb);
end
