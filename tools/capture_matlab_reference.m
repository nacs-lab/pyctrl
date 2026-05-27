function capture_matlab_reference(out_dir)
% capture_matlab_reference  Engine-free capture of MATLAB serialize() output.
%
% Builds the sequences listed in reference_list() and writes, for each:
%   <out_dir>/<name>.bin          int8 bytes of serialize(s)
%   <out_dir>/<name>.params.json  the fixed parameters used
%
% SAFE while an experiment is running -- BUT ONLY IN A SEPARATE MATLAB SESSION:
%   * Calls SeqManager.override_tick_per_sec(1000) so the libnacs engine is
%     never loaded and NO hardware is touched.
%   * Calls serialize() only -- never generate() / run().
%   Do NOT run this in the MATLAB session that is running the experiment: it
%   would share the SeqManager singleton and the base workspace.
%
% The default out_dir is pyctrl/tests/reference/ (byte round-trip references).
% For the engine-accepts check, capture sequences whose channel names match
% config.yml into pyctrl/tests/reference_engine/ instead.

    here = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(out_dir)
        out_dir = fullfile(here, '..', 'tests', 'reference');
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

    specs = reference_list();
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
