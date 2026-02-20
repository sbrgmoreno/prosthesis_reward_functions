function evalLogs = evaluateSAC_Discrete(agentMatPath, numEpisodes)
% evaluateSAC_Discrete
% Evaluates a trained SAC-Discrete actor (greedy on softmax probabilities).
%
% Usage:
%   evalLogs = evaluateSAC_Discrete("C:\...\AgentFinal.mat", 48);
%
% Saves outputs to:
%   C:\QR_DQN\SAC_Discrete\eval_<timestamp>\
%
% Fixes included:
%   - No dependency on configs.RLtrainingOptions
%   - Env created with a valid folder to avoid "\episode00001.mat" permission errors
%   - Saves per-episode figures + returns plot + logs

    if nargin < 2 || isempty(numEpisodes)
        numEpisodes = 48;
    end

    % ---- Project paths
    addpath(genpath('.\src'));
    addpath(genpath('.\config'));
    addpath(genpath('.\lib'));
    addpath(genpath('.\agents'));

    configs = configurables();

    % ---- Output root
    outRoot = "C:\SAC_Discrete";
    if ~exist(outRoot, 'dir'); mkdir(outRoot); end

    timeTag = string(datetime("now","Format","yyMMdd_HHmmss"));
    outDir  = fullfile(outRoot, "SAC_Discrete", "eval_" + timeTag);
    if ~exist(outDir,'dir'); mkdir(outDir); end

    figsDir = fullfile(outDir, "figs");
    if ~exist(figsDir,'dir'); mkdir(figsDir); end

    % IMPORTANT: valid episode folder for Env.saveEpisode()
    episodeOutDir = fullfile(outDir, "episodes");
    if ~exist(episodeOutDir,'dir'); mkdir(episodeOutDir); end

    % ---- Load MAT
    S = load(agentMatPath);

    if isfield(S, "actorNetFinal")
        actorNet = S.actorNetFinal;
    elseif isfield(S, "actorNetToSave")
        actorNet = S.actorNetToSave;
    else
        error("No actor network found in MAT file: %s", agentMatPath);
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

    % ---- Max steps per episode (NO RLtrainingOptions)
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

    % ---- Logs
    evalLogs = struct();
    evalLogs.agentMatPath = agentMatPath;
    evalLogs.numEpisodes  = numEpisodes;
    evalLogs.maxStepsEp   = maxStepsEp;
    evalLogs.numActions   = numActions;
    evalLogs.returns      = zeros(1, numEpisodes);
    evalLogs.actions      = cell(1, numEpisodes);
    evalLogs.outDir       = outDir;

    fprintf("=== SAC-Discrete EVALUATION (greedy) ===\n");
    fprintf("Agent: %s\n", agentMatPath);
    fprintf("Episodes: %d | Steps/Ep: %d\n", numEpisodes, maxStepsEp);
    fprintf("Saving to: %s\n\n", outDir);

    % ---- Evaluation loop
    for ep = 1:numEpisodes
        obs = reset(env);
        epRet = 0;
        actHist = zeros(1, maxStepsEp);

        for t = 1:maxStepsEp
            aIdx = greedyActionFromActor(actorNet, obs, numActions);
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

    % ---- Plot returns
    f = figure('Name', 'SAC-Discrete Evaluation Returns', 'Color', 'w');
    plot(evalLogs.returns, '-o');
    xlabel('Evaluation Episode');
    ylabel('Return');
    title('SAC-Discrete Evaluation Returns (Greedy)');
    grid on;
    drawnow;

    exportgraphics(f, fullfile(figsDir, "SAC_Discrete_Eval_Returns.png"), 'Resolution', 200);
    close(f);

    % ---- Save logs
    save(fullfile(outDir, "SAC_Discrete_Eval_Logs.mat"), "evalLogs");
    fprintf("\nSaved logs: %s\n", fullfile(outDir, "SAC_Discrete_Eval_Logs.mat"));
    fprintf("Saved figures folder: %s\n", figsDir);
    fprintf("Episode logs folder: %s\n", episodeOutDir);
end


function aIdx = greedyActionFromActor(actorNet, obs, numActions)
% Greedy action from actor: softmax(logits) then argmax.
    dlX = dlarray(single(obs(:)), "CB");
    logits = forward(actorNet, dlX); % usually [numActions x 1]

    if isa(logits, "dlarray")
        logits = extractdata(logits);
    end
    logits = single(logits);

    % ensure shape [A x 1]
    logits = reshape(logits, [numActions, 1]);

    p = softmaxStable(logits);
    [~, aIdx] = max(p);
    aIdx = double(aIdx);
end


function p = softmaxStable(logits)
% Stable softmax for column vector
    m = max(logits, [], 1);
    z = logits - m;
    expz = exp(z);
    p = expz ./ sum(expz, 1);
end


function saveEpisodeFigure(env, figsDir, ep)
% Tries env plotters (plot_episode, plot_episode2, plot)
% and saves current figure safely.

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

    fname = fullfile(figsDir, sprintf("SAC_Discrete_Eval_Ep_%03d.png", ep));
    try
        exportgraphics(f, fname, 'Resolution', 200);
    catch
        saveas(f, fname);
    end
    close(f);
end

