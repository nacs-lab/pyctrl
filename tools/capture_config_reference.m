function capture_config_reference(out_json)
% capture_config_reference  Engine-free capture of the real expConfig.m config.
%
% Dumps the three BYTE-LOAD-BEARING parts of SeqConfig -- the channel-alias map,
% the default-value map, and the consts tree -- to a committed JSON file so the
% pyctrl SeqConfig loader (lib/seq_config.py) can resolve `s.C` / `Consts()` reads
% and channel translation byte-identically, WITHOUT a parallel hand-port of the
% 344-line expConfig.m (which drifts as constants get tuned).
%
% ENGINE-FREE BY CONSTRUCTION -- and why we do NOT call SeqConfig.get():
%   SeqConfig's constructor runs `expConfig()` and then, because expConfig sets
%   `configFile = 'config.yml'`, calls SeqManager.load_config_file(configFile) ->
%   SeqManager.get() -> py.eval('Manager()'), which LOADS the libnacs engine.
%   override_tick_per_sec() only gates tick_per_sec(), NOT load_config_file, so it
%   does NOT prevent that load. Loading the engine is compile-side (no hardware),
%   but it (a) is needless for a pure config dump and (b) wedges MATLAB's exit on
%   Windows (libnacs' libzmq static dtor asserts when loaded-but-never-started).
%   So instead we replicate SeqConfig's expConfig ENVIRONMENT directly (predefine
%   the same maps + consts struct + a no-op disableChannel) and run the expConfig
%   script in this function's workspace. No SeqConfig, no SeqManager, no Manager.
%   Channel-alias TRANSLATION (recursive alias expansion) and defaultVals
%   re-keying are done on the Python side (lib/seq_config.py), unit-tested against
%   known cases -- so this script stays pure data extraction.
%
% SAFE while an experiment is running, in a SEPARATE headless session:
%     matlab -batch "addpath('<pyctrl>/tools'); capture_config_reference"
%
%   out_json  destination. Default: pyctrl/tests/reference/config_reference.json

    here = fileparts(mfilename('fullpath'));
    repo = fullfile(here, '..', '..');                 % experiment-control/
    addpath(fullfile(repo, 'matlab_new'));             % for expConfig.m
    if nargin < 1 || isempty(out_json)
        out_json = fullfile(here, '..', 'tests', 'reference', 'config_reference.json');
    end
    odir = fileparts(out_json);
    if ~exist(odir, 'dir'); mkdir(odir); end

    % --- replicate SeqConfig's pre-expConfig environment (SeqConfig.m:47-60) ---
    % WITHOUT the load_config_file branch, so the engine is never loaded.
    channelAlias        = containers.Map();                              %#ok<NASGU>
    defaultVals         = containers.Map();                              %#ok<NASGU>
    niClocks            = containers.Map();
    niStart             = containers.Map();
    m_disabledChannels  = containers.Map('KeyType','char','ValueType','double');
    disableChannel      = @(chn) m_disabledChannels(chn);  %#ok<NASGU>  % never called by expConfig
    consts              = struct();                                      %#ok<NASGU>
    warnUnusedScan      = true;                                          %#ok<NASGU>
    warnUnusedScanFixed = true;                                          %#ok<NASGU>
    configFile          = [];                                            %#ok<NASGU>

    expConfig();   % populates channelAlias / defaultVals / consts in THIS workspace

    % --- trailing-slash trim on alias VALUES (SeqConfig.m:82-85) --------------
    ak = keys(channelAlias);
    av = values(channelAlias);
    for i = 1:numel(av)
        tok = regexp(av{i}, '^(.*[^/])/*$', 'tokens', 'once');
        if ~isempty(tok); av{i} = tok{1}; end
    end

    % --- assemble + write -----------------------------------------------------
    % Maps emitted as parallel key/value arrays (translated-name keys contain '/',
    % which are not valid struct field names, so a struct/jsonencode-of-Map would
    % be lossy/ambiguous). consts is a struct -> nested JSON object.
    out = struct();
    out.channel_alias_keys = ak;                 % cell of alias names
    out.channel_alias_vals = av;                 % cell of (trimmed) backend names
    out.default_vals_keys  = keys(defaultVals);  % cell of alias names (RAW, untranslated)
    out.default_vals_vals  = values(defaultVals);% cell of doubles
    out.consts             = consts;             % resolved nested struct
    out.ni_clocks_keys     = keys(niClocks);     % NI external-clock source per device (PFI)
    out.ni_clocks_vals     = values(niClocks);
    out.ni_start_keys      = keys(niStart);      % NI external start-trigger per device (PFI)
    out.ni_start_vals      = values(niStart);

    json = jsonencode(out);
    fid = fopen(out_json, 'w');
    fwrite(fid, json, 'char');
    fclose(fid);
    fprintf('wrote %s: %d aliases, %d default vals\n', ...
            out_json, numel(ak), numel(out.default_vals_keys));
end
