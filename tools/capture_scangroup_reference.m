function capture_scangroup_reference(out_dir, list_fn)
% capture_scangroup_reference  Engine-free capture of ScanGroup getseq() expansion.
%
% For each ScanGroup returned by a list function, records groupsize, nseq, the run
% parameters, and the FULL ORDERED list of getseq(n) parameter structs, and writes
% them as one JSON object to <out_dir>/scangroup_reference.json. This is the
% ground truth for the Phase-4 W4 expansion-equality oracle
% (pyctrl/tests/test_scan_group_oracle.py).
%
% ScanGroup is the MOST engine-free class in the stack: it references no
% SeqManager / SeqConfig / libnacs / hardware at all, so this is safe to run any
% time -- BUT still run it in a FRESH headless session so it cannot clobber the
% base workspace of a MATLAB that is running the experiment:
%   matlab -batch "cd <pyctrl/tools>; capture_scangroup_reference"
%
% jsonencode preserves struct field insertion order; DO NOT sort keys -- the
% Python oracle compares values (order-insensitive) but the field order is also a
% faithful observable of the merge/expansion algorithm.
%
% Arguments (both optional):
%   out_dir  destination directory. Default: pyctrl/tests/reference_scangroup/
%   list_fn  handle to a list function. Default: @scangroup_list

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(out_dir)
        out_dir = fullfile(here, '..', 'tests', 'reference_scangroup');
    end
    if nargin < 2 || isempty(list_fn)
        list_fn = @scangroup_list;
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    repo = fullfile(here, '..', '..');                  % experiment-control/
    addpath(fullfile(repo, 'matlab_new', 'lib'));
    addpath(fullfile(repo, 'matlab_new'));

    specs = list_fn();
    out = struct();
    for i = 1:numel(specs)
        name = specs(i).name;
        g = specs(i).build();
        ns = nseq(g);
        seqs = cell(1, ns);
        for n = 1:ns
            seqs{n} = getseq(g, n);
        end
        rp = g.runp();                                  % a DynProps...
        entry = struct();
        entry.groupsize = groupsize(g);
        entry.nseq = ns;
        entry.runp = rp();                              % ...resolved to a struct
        entry.seqs = seqs;                              % cell -> JSON array of objects
        out.(name) = entry;
        fprintf('captured %s: groupsize=%d nseq=%d\n', name, groupsize(g), ns);
    end

    fid = fopen(fullfile(out_dir, 'scangroup_reference.json'), 'w');
    fwrite(fid, jsonencode(out), 'char');
    fclose(fid);
    fprintf('done: %d scan groups -> %s\n', numel(specs), ...
            fullfile(out_dir, 'scangroup_reference.json'));
end
