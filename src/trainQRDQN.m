function logs = trainQRDQN(agent_id)
% trainQRDQN: custom training loop for QR-DQN in your Env
% agent_id is just for naming/logging (optional)

% --- Paths (match your trainInterface style)
addpath(genpath('.\src'))
addpath(genpath('.\config'))
addpath(genpath('.\lib'))
addpath(genpath('.\agents'))

configs = configurables();

%% Specs
obsInfo = Env.defineObservationInfo();
actInfo = Env.defineActionDiscreteInfo();
actionsSet = actInfo.Elements;     % cell of 3 vectors
numActions = numel(actionsSet);
obsDim = prod(obsInfo.Dimension);

%% Hyperparameters (start stable)
numQuantiles = 51;
gamma = 0.97;

maxEpisodes = configs.RLtrainingOptions.MaxEpisodes;         % 5000
maxStepsEp  = configs.RLtrainingOptions.MaxStepsPerEpisode;  % 50

batchSize = 64;
bufferCapacity = 200000;
warmupSteps = 2000;

learnRate = 1e-4;
targetUpdateSteps = 1000; % hard update every N steps

% epsilon schedule (per-step)
epsilon = 1.0;
epsilonMin = 0.001;
epsilonDecay = 1.5e-4;

% Quantile Huber
kappa = 1.0;

% Save/plot frequencies
saveEveryEpisodes = 500;   % checkpoints like SaveAgentValue=500
plotEveryEpisodes = 10;    % live plot update

%% Create save directory (like trainInterface)
agent_dir = "";
if configs.flagSaveTraining
    agent_dir = fullfile("C:\", "trainedAgentsProtesisNew", ...
        agent_id, "_", string(datetime("now","Format","yy-MM-dd HH mm ss")));
    mkdir(agent_dir);
    save(fullfile(agent_dir, "00_configs.mat"), "configs", "agent_id");
end

%% Environment (same logic as trainInterface)
if configs.usePrerecorded
    [emg, glove] = getDataset(configs.dataset, configs.dataset_folder);
    env = Env(agent_dir, true, emg, glove);
else
    env = Env(agent_dir, false);
end

%% Networks
onlineNet = agent_01_qrdqn(obsInfo, actInfo, numQuantiles);
targetNet = onlineNet;

%% Optimizer state (Adam)
avgGrad = [];
avgSqGrad = [];
globalStep = 0;

%% Replay buffer
rb = replayBuffer(bufferCapacity, obsDim);

%% Logs
logs.agent_id = agent_id;
logs.episodeReturn = zeros(1, maxEpisodes);
logs.avg100 = zeros(1, maxEpisodes);
logs.epsilon = zeros(1, maxEpisodes);
logs.globalStepEnd = zeros(1, maxEpisodes);
logs.loss = nan(1, maxEpisodes); % average loss per episode (optional)

%% Figure setup
fig = figure(1);
set(fig, 'Name', 'QR-DQN Training', 'NumberTitle', 'off');

for ep = 1:maxEpisodes
    obs = reset(env);
    epRet = 0;
    epLossSum = 0;
    epLossCount = 0;

    for t = 1:maxStepsEp
        globalStep = globalStep + 1;

        % --- epsilon-greedy action selection (index 1..numActions)
        if rand < epsilon || rb.count < warmupSteps
            aIdx = randi(numActions);
        else
            aIdx = selectActionQR(onlineNet, obs, numActions, numQuantiles);
        end

        % --- map index -> actual action vector expected by Env.step
        actionVec = actionsSet{aIdx};

        [nextObs, r, done, ~] = step(env, actionVec);

        epRet = epRet + r;

        % store (obs, actionIndex, reward, nextObs, done)
        rb.add(obs, aIdx, r, nextObs, done);

        obs = nextObs;

        % --- learning step
        if rb.count >= warmupSteps
            [S, A, R, S2, D] = rb.sample(batchSize);

            dlS  = dlarray(single(S), "CB");
            dlS2 = dlarray(single(S2), "CB");

            [lossVal, grads] = dlfeval(@qrDqnGradients, onlineNet, targetNet, ...
                dlS, A, R, dlS2, D, gamma, numActions, numQuantiles, kappa);

            [onlineNet, avgGrad, avgSqGrad] = adamupdate( ...
                onlineNet, grads, avgGrad, avgSqGrad, globalStep, learnRate);

            % accumulate loss for episode logging
            epLossSum = epLossSum + double(gather(extractdata(lossVal)));
            epLossCount = epLossCount + 1;

            % target update
            if mod(globalStep, targetUpdateSteps) == 0
                targetNet = onlineNet;
            end
        end

        % epsilon decay per step
        epsilon = max(epsilonMin, epsilon - epsilonDecay);

        if done
            break;
        end
    end

    logs.episodeReturn(ep) = epRet;
    logs.avg100(ep) = mean(logs.episodeReturn(max(1,ep-99):ep));
    logs.epsilon(ep) = epsilon;
    logs.globalStepEnd(ep) = globalStep;
    if epLossCount > 0
        logs.loss(ep) = epLossSum / epLossCount;
    end

    fprintf("Ep %d | Return %.4f | Avg100 %.4f | eps %.4f | steps %d\n", ...
        ep, logs.episodeReturn(ep), logs.avg100(ep), epsilon, globalStep);

    % --- live plot
    if mod(ep, plotEveryEpisodes) == 0
        figure(fig); clf;

        plot(logs.episodeReturn(1:ep), 'Color', [0.7 0.7 0.7]); hold on;
        plot(logs.avg100(1:ep), 'LineWidth', 2);
        xlabel('Episode'); ylabel('Return');
        title(sprintf("QR-DQN Training | %s | Ep %d", agent_id, ep));
        legend('Episode Return', 'Avg(100)', 'Location', 'best');
        grid on;
        drawnow;

        if configs.flagSaveTraining && agent_dir ~= ""
            saveas(fig, fullfile(agent_dir, "training_curve_live.png"));
        end
    end

    % --- checkpoint save
    if configs.flagSaveTraining && agent_dir ~= "" && mod(ep, saveEveryEpisodes) == 0
        onlineNetToSave = onlineNet; %#ok<NASGU>
        targetNetToSave = targetNet; %#ok<NASGU>
        epsilonToSave = epsilon; %#ok<NASGU>
        numQuantilesToSave = numQuantiles; %#ok<NASGU>
        gammaToSave = gamma; %#ok<NASGU>
        save(fullfile(agent_dir, sprintf("AgentEp_%d.mat", ep)), ...
            "onlineNetToSave", "targetNetToSave", ...
            "ep", "epsilonToSave", "numQuantilesToSave", "gammaToSave", ...
            "logs");
    end

end

% --- final save
if configs.flagSaveTraining && agent_dir ~= ""
    onlineNetFinal = onlineNet; %#ok<NASGU>
    targetNetFinal = targetNet; %#ok<NASGU>
    save(fullfile(agent_dir, "AgentFinal.mat"), ...
        "onlineNetFinal", "targetNetFinal", "logs", "numQuantiles", "gamma");
end

end

%% ===================== Helper functions =====================

function aIdx = selectActionQR(net, obs, numActions, N)
% Choose action by maximizing mean of predicted quantiles.
dlX = dlarray(single(obs(:)), "CB");
Z = forward(net, dlX);                 % [(A*N) x 1]
Z = extractdata(Z);
Z = reshape(Z, [N, numActions]);       % [N x A]
Q = mean(Z, 1);                        % [1 x A]
[~, aIdx] = max(Q);
aIdx = double(aIdx);
end

function [loss, grads] = qrDqnGradients(onlineNet, targetNet, dlS, A, R, dlS2, D, gamma, numActions, N, kappa)
% Quantile Huber Loss for QR-DQN + Double DQN action selection.
% This version avoids breaking dlarray tracing (NO numeric assignment loops).

B = size(dlS, 2);

% --------- Online quantiles at S (all actions)
Zall = forward(onlineNet, dlS);                 % [(A*N) x B] dlarray
Zall = reshape(Zall, [N, numActions, B]);       % [N x A x B]

% --------- Select Z(s,a) using one-hot mask (keeps tracing)
% A is [1 x B] with values 1..numActions
maskA = zeros(numActions, B, 'single');
maskA(sub2ind([numActions, B], A, 1:B)) = 1;
dlMaskA = dlarray(maskA);                       % [A x B]
dlMaskA = reshape(dlMaskA, [1, numActions, B]); % [1 x A x B]

Zsa = sum(Zall .* dlMaskA, 2);                  % [N x 1 x B]
Zsa = squeeze(Zsa);                             % [N x B] dlarray

% --------- Double DQN: choose a* at S2 using ONLINE net (mean over quantiles)
Z2_online = forward(onlineNet, dlS2);
Z2_online = reshape(Z2_online, [N, numActions, B]);
Q2 = squeeze(mean(Z2_online, 1));               % [A x B] dlarray

% argmax works fine here; aStar is numeric indices (not differentiated)
[~, aStar] = max(gather(extractdata(Q2)), [], 1);    % [1 x B] numeric

% --------- Target quantiles at S2 using TARGET net, action aStar
Z2_target = forward(targetNet, dlS2);
Z2_target = reshape(Z2_target, [N, numActions, B]);

maskStar = zeros(numActions, B, 'single');
maskStar(sub2ind([numActions, B], aStar, 1:B)) = 1;
dlMaskStar = dlarray(maskStar);
dlMaskStar = reshape(dlMaskStar, [1, numActions, B]);

Ztarget = sum(Z2_target .* dlMaskStar, 2);      % [N x 1 x B]
Ztarget = squeeze(Ztarget);                     % [N x B] dlarray

% --------- 1-step target quantiles (luego podrás cambiar a n-step)
T = single(R) + gamma .* (1 - single(D)) .* Ztarget;   % [N x B] dlarray
% Nota: R y D vienen como numeric; al operar con dlarray se promueven bien.

% --------- Quantile Huber loss (pairwise)
taus = ((1:N) - 0.5) ./ N;                      % [1 x N]
taus = reshape(single(taus), [N 1]);            % [N x 1]
dlTaus = dlarray(taus);                         % [N x 1]

% delta_ij = T_j - Z_i
Texp = reshape(T,   [1 N B]);                   % [1 x N x B]
Zexp = reshape(Zsa, [N 1 B]);                   % [N x 1 x B]
delta = Texp - Zexp;                            % [N x N x B]

absDelta = abs(delta);
huber = (absDelta <= kappa).*0.5.*delta.^2 + (absDelta > kappa).*kappa.*(absDelta - 0.5*kappa);

ind = delta < 0;                                % logical dlarray
tauMat = repmat(dlTaus, 1, N);                  % [N x N]
tauMat = reshape(tauMat, [N N 1]);              % [N x N x 1]

w = abs(tauMat - single(ind));                  % [N x N x B]

loss = mean(w .* huber, "all");                 % scalar dlarray traced

% --------- Gradients (works now)
grads = dlgradient(loss, onlineNet.Learnables);
end