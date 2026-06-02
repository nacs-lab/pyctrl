function capture_scan_point_reference(out_json, list_fn)
% capture_scan_point_reference  Engine-free per-point byte capture for the W6 oracle.
%
% For each (scan, sequence) pair from scan_point_list, expands the ScanGroup point by
% point and serializes the sequence built for each point -- exactly the RunScans inner
% loop -- writing the per-point bytes (hex) to <...>/scan_point_reference.json:
%   { "<name>": { "seq": "<SeqFn>", "nseq": N, "points": ["<hex>", ...] }, ... }
%
% SAFETY (same as capture_ybseqs_reference): serialize() never runs the deferred
% regBeforeStart/regAfterEnd callbacks (camera/AWG/server/MemoryMap), and
% override_tick_per_sec keeps the libnacs engine unloaded. Real sequence code, so run it
% in a fresh headless `matlab -batch` and preferably a maintenance window:
%   matlab -batch "cd <pyctrl/tools>; capture_scan_point_reference"

    here = fileparts(mfilename('fullpath'));
    repo = fullfile(here, '..', '..');
    mn = fullfile(repo, 'matlab_new');
    if nargin < 1 || isempty(out_json)
        out_json = fullfile(here, '..', 'tests', 'reference_scan_point', ...
                            'scan_point_reference.json');
    end
    if nargin < 2 || isempty(list_fn)
        list_fn = @scan_point_list;
    end
    odir = fileparts(out_json);
    if ~exist(odir, 'dir'), mkdir(odir); end

    for d = {mn, fullfile(mn,'lib'), fullfile(mn,'YbSeqs'), fullfile(mn,'YbSteps'), ...
             fullfile(mn,'YbScans'), fullfile(mn,'YbScans','scanConfig'), ...
             fullfile(mn,'YbRearrangement'), fullfile(mn,'YbFPGAWaveforms'), ...
             fullfile(mn,'YbExptCtrl')}
        if exist(d{1}, 'dir'), addpath(d{1}); end
    end

    % Real production tick rate (1e12 == 1 ps): us-scale steps round to 0 ticks at the
    % default 1000/s and are rejected. Engine stays unloaded.
    SeqManager.override_tick_per_sec(1e12);
    cleanup = onCleanup(@() SeqManager.override_tick_per_sec(0)); %#ok<NASGU>

    specs = list_fn();
    out = struct();
    for i = 1:numel(specs)
        name = specs(i).name;
        g = specs(i).build();
        seqfn = str2func(specs(i).seq);
        ns = nseq(g);
        pts = cell(1, ns);
        for n = 1:ns
            params = getseq(g, n);              % resolved scalar params for point n
            s = ExpSeq(params);                 % merge params onto consts
            s = seqfn(s);                       % build the step tree (no callbacks run)
            b = serialize(s);                   % NO generate()/run()
            pts{n} = lower(sprintf('%02x', typecast(int8(b), 'uint8')));
        end
        entry = struct();
        entry.seq = specs(i).seq;
        entry.nseq = ns;
        entry.points = pts;                     % cell of hex -> JSON array of strings
        out.(name) = entry;
        fprintf('captured %s (%s): %d points\n', name, specs(i).seq, ns);
    end

    fid = fopen(out_json, 'w');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
    fprintf('done: %d (scan, seq) pairs -> %s\n', numel(specs), out_json);
end
