function evalLogs = evaluateQRDQN(agentMatPath, numEpisodes)
% evaluateQRDQN
% Evaluate a trained QR-DQN (saved as dlnetwork) with greedy policy.
%
% Usage:
%   evalLogs = evaluateQRDQN("C:\...\AgentFinal.mat", 48);
%
% Saves outputs to:
%   C:\QR_DQN\QRDQN\eval_<timestamp>\
%
% This version FIXES the permission denied error by ensuring Env receives a
% valid episode folder (so Env.saveEpisode doesn't try to write to \episodeXXXX.mat).

    if nargin < 2 || isempty(numEpisodes)
        numEpisodes = 48;
    end

    % ---- Project paths
    addpath(genpath('.\src'));
    addpath(genpath('.\config'));
    addpath(genpath('.\lib'));
    addpath(genpath('.\agents'));

    configs = configurables();

    % ---- Output root (requested)
    outRoot = "C:\QR_DQN";
    if ~exist(outRoot, 'dir'); mkdir(outRoot); end

    timeTag = string(datetime("now","Format","yyMMdd_HHmmss"));
    outDir  = fullfile(outRoot, "QRDQN", "eval_" + timeTag);
    if ~exist(outDir,'dir'); mkdir(outDir); end

    figsDir = fullfile(outDir, "figs");
    if ~exist(figsDir,'dir'); mkdir(figsDir); end

    % IMPORTANT: give Env a valid folder so it can save episode logs safely
    episodeOutDir = fullfile(outDir, "episodes");
    if ~exist(episodeOutDir,'dir'); mkdir(episodeOutDir); end

    % ---- Load MAT contents
    S = load(agentMatPath);

    % ---- Extract online network
    if isfield(S, "onlineNetFinal")
        onlineNet = S.onlineNetFinal;
    elseif isfield(S, "onlineNetToSave")
        onlineNet = S.onlineNetToSave;
    else
        error("No online network found in MAT file: %s", agentMatPath);
    end

    % ---- Quantiles count
    if isfield(S, "numQuantiles")
        numQuantiles = S.numQuantiles;
    elseif isfield(S, "numQuantilesToSave")
        numQuantiles = S.numQuantilesToSave;
    else
        numQuantiles = 51;
    end

    % ---- Specs
    actInfo = Env.defineActionDiscreteInfo();
    actionsSet = actInfo.Elements;
    numActions = numel(actionsSet);

    % ---- Env (prerecorded vs real)
    if isfield(configs, "usePrerecorded") && configs.usePrerecorded
        [emg, glove] = getDataset(configs.dataset, configs.dataset_folder);
        env = Env(episodeOutDir, true, emg, glove);
    else
        env = Env(episodeOutDir, false);
    end

    % ---- Max steps per episode (DO NOT use RLtrainingOptions)
    maxStepsEp = 50; % fallback
    if isfield(configs, "maxNumberStepsInEpisodes") && ~isempty(configs.maxNumberStepsInEpisodes)
        maxStepsEp = configs.maxNumberStepsInEpisodes;
    elseif isfield(configs, "simOpts")
        try
            if isprop(configs.simOpts, "MaxSteps") && ~isempty(configs.simOpts.MaxSteps)
                maxStepsEp = configs.simOpts.MaxSteps;
            end
        catch
            % ignore
        end
    end

    % ---- Prepare logs
    evalLogs = struct();
    evalLogs.agentMatPath  = agentMatPath;
    evalLogs.numEpisodes   = numEpisodes;
    evalLogs.numQuantiles  = numQuantiles;
    evalLogs.numActions    = numActions;
    evalLogs.maxStepsEp    = maxStepsEp;
    evalLogs.returns       = zeros(1, numEpisodes);
    evalLogs.actions       = cell(1, numEpisodes);
    evalLogs.outDir        = outDir;

    fprintf("=== QR-DQN EVALUATION (greedy) ===\n");
    fprintf("Agent: %s\n", agentMatPath);
    fprintf("Episodes: %d | Steps/Ep: %d | Quantiles: %d\n", numEpisodes, maxStepsEp, numQuantiles);
    fprintf("Saving to: %s\n\n", outDir);

    % ---- Evaluation loop
    for ep = 1:numEpisodes
        obs = reset(env);
        epRet = 0;
        actHist = zeros(1, maxStepsEp);

        for t = 1:maxStepsEp
            aIdx = selectActionQR(onlineNet, obs, numActions, numQuantiles);
            actionVec = actionsSet{aIdx};

            [nextObs, r, done, ~] = step(env, actionVec);

            epRet = epRet + r;
            actHist(t) = aIdx;
            obs = nextObs;

            if done
                actHist = actHist(1:t);
                break;
            end
        end

        evalLogs.returns(ep) = epRet;
        evalLogs.actions{ep} = actHist;

        fprintf("Eval Ep %03d | Return = %.6f | Steps = %d\n", ep, epRet, numel(actHist));

        % ---- Save per-episode figure (env plots)
        saveEpisodeFigure(env, figsDir, ep);
    end

    % ---- Plot and save returns figure
    f = figure('Name', 'QR-DQN Evaluation Returns', 'Color', 'w');
    plot(evalLogs.returns, '-o');
    xlabel('Evaluation Episode');
    ylabel('Return');
    title('QR-DQN Evaluation Returns (Greedy)');
    grid on;
    drawnow;

    exportgraphics(f, fullfile(figsDir, "QRDQN_Eval_Returns.png"), 'Resolution', 200);
    close(f);

    % ---- Save logs
    save(fullfile(outDir, "QRDQN_Eval_Logs.mat"), "evalLogs");
    fprintf("\nSaved logs: %s\n", fullfile(outDir, "QRDQN_Eval_Logs.mat"));
    fprintf("Saved figures folder: %s\n", figsDir);
    fprintf("Episode logs folder: %s\n", episodeOutDir);
end


function aIdx = selectActionQR(net, obs, numActions, N)
% Greedy action selection: mean over quantiles per action, then argmax.
    dlX = dlarray(single(obs(:)), "CB"); % column batch
    Z = forward(net, dlX);

    if isa(Z, "dlarray")
        Z = extractdata(Z);
    end

    Z = reshape(Z, [N, numActions]); % [quantiles, actions]
    Q = mean(Z, 1);                  % expected value per action
    [~, aIdx] = max(Q);
    aIdx = double(aIdx);
end


function saveEpisodeFigure(env, figsDir, ep)
% Tries to call env plotters (plot_episode, plot_episode2, plot)
% and saves the current figure. Closes only the produced figure.

    didPlot = false;

    try
        if any(strcmp(methods(env), "plot_episode"))
            env.plot_episode;
            didPlot = true;
        elseif any(strcmp(methods(env), "plot_episode2"))
            env.plot_episode2;
            didPlot = true;
        elseif any(strcmp(methods(env), "plot"))
            env.plot;
            didPlot = true;
        end
    catch
        didPlot = false;
    end

    if ~didPlot
        return;
    end

    drawnow;
    f = gcf;
    fname = fullfile(figsDir, sprintf("QRDQN_Eval_Ep_%03d.png", ep));

    try
        exportgraphics(f, fname, 'Resolution', 200);
    catch
        saveas(f, fname);
    end

    close(f);
end

