function capture_matlab_reference(out_dir, list_fn)
% capture_matlab_reference  Engine-free capture of MATLAB serialize() output.
%
% Builds the sequences returned by a reference-list function and writes, for each:
%   <out_dir>/<name>.bin          int8 bytes of serialize(s)
%   <out_dir>/<name>.params.json  the fixed parameters used
%
% SAFE while an experiment is running -- BUT ONLY IN A SEPARATE MATLAB SESSION:
%   * Calls SeqManager.override_tick_per_sec(1000) so the libnacs engine is
%     never loaded and NO hardware is touched.
%   * Calls serialize() only -- never generate() / run().
%   Do NOT run this in the MATLAB session that is running the experiment: it
%   would share the SeqManager singleton and the base workspace. The simplest
%   safe way to run it is a fresh headless session:
%     matlab -batch "cd <here>; capture_matlab_reference"
%
% Arguments (both optional):
%   out_dir  destination directory. Default: pyctrl/tests/reference/
%   list_fn  handle to a reference-list function. Default: @reference_list
%
% Two standard captures:
%   capture_matlab_reference                                   % byte round-trip refs
%   capture_matlab_reference(fullfile('..','tests','reference_engine'), ...
%                            @reference_list_engine)           % engine-accepts refs
% (the engine refs use config.yml channel names; the round-trip refs use
% 'Device/CH' placeholders).

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(out_dir)
        out_dir = fullfile(here, '..', 'tests', 'reference');
    end
    if nargin < 2 || isempty(list_fn)
        list_fn = @reference_list;
    end
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % --- add the MATLAB framework to the path (lib + matlab_new) ---
    repo = fullfile(here, '..', '..');                  % experiment-control/
    addpath(fullfile(repo, 'matlab_new', 'lib'));
    addpath(fullfile(repo, 'matlab_new'));

    % --- engine-free mode: fixed tick rate => no libnacs load, no hardware ---
    SeqManager.override_tick_per_sec(1000);
    cleanup = onCleanup(@() SeqManager.override_tick_per_sec(0)); %#ok<NASGU>

    specs = list_fn();
    for i = 1:numel(specs)
        name = specs(i).name;
        s = specs(i).build();
        bytes = serialize(s);                           % NO generate()/run()

        fid = fopen(fullfile(out_dir, [name '.bin']), 'w');
        fwrite(fid, bytes, 'int8');
        fclose(fid);

        fid = fopen(fullfile(out_dir, [name '.params.json']), 'w');
        fwrite(fid, jsonencode(specs(i).params), 'char');
        fclose(fid);

        fprintf('wrote %s.bin (%d bytes)\n', name, numel(bytes));
    end
    fprintf('done: %d references written to %s\n', numel(specs), out_dir);
end
