function [logs, agent_dir] = trainSAC_Discrete(agent_id)
% trainSAC_Discrete
% Discrete Soft Actor-Critic (SAC) with fixed alpha.
% - live training plot
% - checkpoints + final save

%% Paths
addpath(genpath('.\src'))
addpath(genpath('.\config'))
addpath(genpath('.\lib'))
addpath(genpath('.\agents'))

configs = configurables();

%% Specs
obsInfo = Env.defineObservationInfo();
actInfo = Env.defineActionDiscreteInfo();
actionsSet = actInfo.Elements;   % cell of 3 vectors
numActions = numel(actionsSet);
obsDim = prod(obsInfo.Dimension);

%% Hyperparameters (buenos defaults para tu caso)
gamma = 0.97;

alpha = 0.2;          % <-- FIXED alpha (pedido)
tau = 0.005;          % Polyak update

maxEpisodes = configs.RLtrainingOptions.MaxEpisodes;          % 5000
maxStepsEp  = configs.RLtrainingOptions.MaxStepsPerEpisode;   % 50

batchSize = 64;
bufferCapacity = 200000;
warmupSteps = 2000;

lrActor  = 3e-4;
lrCritic = 3e-4;

updatesPerStep = 1;   % 1 update por paso (puedes subir a 2 después)
plotEveryEpisodes = 10;
saveEveryEpisodes = 500;

%% Create save directory (similar style)
agent_dir = "";
if configs.flagSaveTraining
    agent_dir = fullfile("C:\", "trainedAgentsProtesisNew", ...
        agent_id, "_", string(datetime("now","Format","yy-MM-dd HH mm ss")));
    mkdir(agent_dir);
    save(fullfile(agent_dir, "00_configs.mat"), "configs", "agent_id", "alpha", "tau", "gamma");
end

%% Env (igual que trainInterface)
if configs.usePrerecorded
    [emg, glove] = getDataset(configs.dataset, configs.dataset_folder);
    env = Env(agent_dir, true, emg, glove);
else
    env = Env(agent_dir, false);
end

%% Networks
actorNet  = agent_sac_actor_discrete(obsInfo, actInfo);

critic1Net = agent_sac_critic_discrete(obsInfo, actInfo);
critic2Net = agent_sac_critic_discrete(obsInfo, actInfo);

targetCritic1Net = critic1Net;
targetCritic2Net = critic2Net;

%% Optimizer states (Adam)
avgGradA = []; avgSqGradA = [];
avgGradC1 = []; avgSqGradC1 = [];
avgGradC2 = []; avgSqGradC2 = [];

globalStep = 0;

%% Replay
rb = replayBuffer(bufferCapacity, obsDim);

%% Logs
logs.agent_id = agent_id;
logs.episodeReturn = zeros(1, maxEpisodes);
logs.avg100 = zeros(1, maxEpisodes);
logs.actorLoss = nan(1, maxEpisodes);
logs.criticLoss = nan(1, maxEpisodes);

%% Figure
fig = figure(1);
set(fig, 'Name', 'SAC-Discrete Training', 'NumberTitle', 'off');

for ep = 1:maxEpisodes
    obs = reset(env);
    epRet = 0;

    epActorLossSum = 0; epActorLossCount = 0;
    epCriticLossSum = 0; epCriticLossCount = 0;

    for t = 1:maxStepsEp
        globalStep = globalStep + 1;

        % -------- Select action (sample from actor policy) or random warmup
        if rb.count < warmupSteps
            aIdx = randi(numActions);
        else
            aIdx = sampleActionFromActor(actorNet, obs, numActions);
        end

        % Map index -> real action vector for Env.step
        actionVec = actionsSet{aIdx};
        [nextObs, r, done, ~] = step(env, actionVec);

        epRet = epRet + r;

        % Store in replay
        rb.add(obs, aIdx, r, nextObs, done);
        obs = nextObs;

        % -------- Update networks
        if rb.count >= warmupSteps
            for u = 1:updatesPerStep
                [S, A, R, S2, D] = rb.sample(batchSize);

                dlS  = dlarray(single(S), "CB");
                dlS2 = dlarray(single(S2), "CB");

                % --- Critic gradients
                [cLoss, gradsC1, gradsC2] = dlfeval(@criticGradientsSAC, ...
                    critic1Net, critic2Net, targetCritic1Net, targetCritic2Net, ...
                    actorNet, dlS, A, R, dlS2, D, gamma, alpha, numActions);

                [critic1Net, avgGradC1, avgSqGradC1] = adamupdate(critic1Net, gradsC1, ...
                    avgGradC1, avgSqGradC1, globalStep, lrCritic);
                [critic2Net, avgGradC2, avgSqGradC2] = adamupdate(critic2Net, gradsC2, ...
                    avgGradC2, avgSqGradC2, globalStep, lrCritic);

                epCriticLossSum = epCriticLossSum + double(gather(extractdata(cLoss)));
                epCriticLossCount = epCriticLossCount + 1;

                % --- Actor gradients (uses updated critics)
                [aLoss, gradsA] = dlfeval(@actorGradientsSAC, ...
                    actorNet, critic1Net, critic2Net, dlS, alpha, numActions);

                [actorNet, avgGradA, avgSqGradA] = adamupdate(actorNet, gradsA, ...
                    avgGradA, avgSqGradA, globalStep, lrActor);

                epActorLossSum = epActorLossSum + double(gather(extractdata(aLoss)));
                epActorLossCount = epActorLossCount + 1;

                % --- Soft update target critics (R2023b safe)
                targetCritic1Net = softUpdateDlnet(targetCritic1Net, critic1Net, tau);
                targetCritic2Net = softUpdateDlnet(targetCritic2Net, critic2Net, tau);

            end
        end

        if done
            break;
        end
    end

    logs.episodeReturn(ep) = epRet;
    logs.avg100(ep) = mean(logs.episodeReturn(max(1,ep-99):ep));

    if epActorLossCount > 0
        logs.actorLoss(ep) = epActorLossSum / epActorLossCount;
    end
    if epCriticLossCount > 0
        logs.criticLoss(ep) = epCriticLossSum / epCriticLossCount;
    end

    fprintf("Ep %d | Return %.4f | Avg100 %.4f | ActorLoss %.4f | CriticLoss %.4f\n", ...
        ep, logs.episodeReturn(ep), logs.avg100(ep), logs.actorLoss(ep), logs.criticLoss(ep));

    % ---- live plot
    if mod(ep, plotEveryEpisodes) == 0
        figure(fig); clf;
        plot(logs.episodeReturn(1:ep), 'Color', [0.7 0.7 0.7]); hold on;
        plot(logs.avg100(1:ep), 'LineWidth', 2);
        xlabel('Episode'); ylabel('Return');
        title(sprintf("SAC-Discrete | %s | Ep %d", agent_id, ep));
        legend('Episode Return', 'Avg(100)', 'Location', 'best');
        grid on;
        drawnow;

        if configs.flagSaveTraining && agent_dir ~= ""
            saveas(fig, fullfile(agent_dir, "training_curve_live.png"));
        end
    end

    % ---- checkpoint save
    if configs.flagSaveTraining && agent_dir ~= "" && mod(ep, saveEveryEpisodes) == 0
        actorNetToSave = actorNet; %#ok<NASGU>
        critic1NetToSave = critic1Net; %#ok<NASGU>
        critic2NetToSave = critic2Net; %#ok<NASGU>
        targetCritic1NetToSave = targetCritic1Net; %#ok<NASGU>
        targetCritic2NetToSave = targetCritic2Net; %#ok<NASGU>
        save(fullfile(agent_dir, sprintf("AgentEp_%d.mat", ep)), ...
            "actorNetToSave", "critic1NetToSave", "critic2NetToSave", ...
            "targetCritic1NetToSave", "targetCritic2NetToSave", ...
            "alpha", "tau", "gamma", "ep", "logs");
    end
end

% ---- final save
if configs.flagSaveTraining && agent_dir ~= ""
    actorNetFinal = actorNet; %#ok<NASGU>
    critic1NetFinal = critic1Net; %#ok<NASGU>
    critic2NetFinal = critic2Net; %#ok<NASGU>
    targetCritic1NetFinal = targetCritic1Net; %#ok<NASGU>
    targetCritic2NetFinal = targetCritic2Net; %#ok<NASGU>
    save(fullfile(agent_dir, "AgentFinal.mat"), ...
        "actorNetFinal", "critic1NetFinal", "critic2NetFinal", ...
        "targetCritic1NetFinal", "targetCritic2NetFinal", ...
        "alpha", "tau", "gamma", "logs");
end

end

%% ===================== Helper functions =====================

function aIdx = sampleActionFromActor(actorNet, obs, numActions)
% Sample action index from pi(a|s) returned by actor (categorical sampling).
dlX = dlarray(single(obs(:)), "CB");
logits = forward(actorNet, dlX);           % [A x 1] dlarray
[pi, ~] = softmaxAndLogSoftmax(logits);    % [A x 1] dlarray

p = gather(extractdata(pi));              % numeric probs
p = max(p, 0); p = p ./ sum(p);           % safety normalize

u = rand;
cdf = cumsum(p);
aIdx = find(u <= cdf, 1, 'first');
if isempty(aIdx), aIdx = randi(numActions); end
aIdx = double(aIdx);
end

function [pi, logPi] = softmaxAndLogSoftmax(logits)
% Stable softmax and log-softmax for dlarray.
% logits: [A x B] or [A x 1]
m = max(logits, [], 1);
z = logits - m;
expz = exp(z);
sumexp = sum(expz, 1);
pi = expz ./ sumexp;
logPi = z - log(sumexp);
end

function [criticLoss, gradsC1, gradsC2] = criticGradientsSAC( ...
    critic1Net, critic2Net, targetCritic1Net, targetCritic2Net, ...
    actorNet, dlS, A, R, dlS2, D, gamma, alpha, numActions)

B = size(dlS, 2);

% --- Current Q(s,·)
Q1_all = forward(critic1Net, dlS);  % [A x B]
Q2_all = forward(critic2Net, dlS);  % [A x B]

% --- Select Q(s,a) with one-hot mask (keeps tracing)
maskA = zeros(numActions, B, 'single');
maskA(sub2ind([numActions, B], A, 1:B)) = 1;
dlMaskA = dlarray(maskA);

Q1_sa = sum(Q1_all .* dlMaskA, 1); % [1 x B]
Q2_sa = sum(Q2_all .* dlMaskA, 1); % [1 x B]

% --- Next policy pi(a|s') from actor
logits2 = forward(actorNet, dlS2);         % [A x B]
[pi2, logPi2] = softmaxAndLogSoftmax(logits2);

% --- Target critics Q(s',·)
Q1t_all = forward(targetCritic1Net, dlS2);
Q2t_all = forward(targetCritic2Net, dlS2);
Qmin_t = min(Q1t_all, Q2t_all);            % [A x B]

% --- Soft value for discrete SAC:
% V(s') = sum_a pi(a|s') * (Qmin(s',a) - alpha*log pi(a|s'))
V2 = sum(pi2 .* (Qmin_t - alpha .* logPi2), 1);  % [1 x B]

% --- Target y (treat as constant w.r.t critic params; that’s fine)
y = single(R) + gamma .* (1 - single(D)) .* V2;  % [1 x B]

% --- Critic losses (MSE)
c1 = Q1_sa - y;
c2 = Q2_sa - y;

criticLoss = mean(c1.^2 + c2.^2, "all");

gradsC1 = dlgradient(criticLoss, critic1Net.Learnables);
gradsC2 = dlgradient(criticLoss, critic2Net.Learnables);
end

function [actorLoss, gradsA] = actorGradientsSAC(actorNet, critic1Net, critic2Net, dlS, alpha, numActions)
% Actor loss:
% E_s [ sum_a pi(a|s) ( alpha log pi(a|s) - min(Q1,Q2)(s,a) ) ]

logits = forward(actorNet, dlS);             % [A x B]
[pi, logPi] = softmaxAndLogSoftmax(logits);  % [A x B]

Q1_all = forward(critic1Net, dlS);           % [A x B]
Q2_all = forward(critic2Net, dlS);           % [A x B]
Qmin = min(Q1_all, Q2_all);                  % [A x B]

actorLoss = mean(sum(pi .* (alpha .* logPi - Qmin), 1), "all"); % scalar

gradsA = dlgradient(actorLoss, actorNet.Learnables);
end

function targetNet = softUpdateDlnet(targetNet, sourceNet, tau)
% Polyak averaging: target = (1-tau)*target + tau*source
% Compatible with MATLAB R2023b (avoids dlupdate with multiple dlnetwork inputs)

tLearn = targetNet.Learnables;
sLearn = sourceNet.Learnables;

% Sanity check: same number/order of learnables
if height(tLearn) ~= height(sLearn)
    error("softUpdateDlnet: Learnables size mismatch (target=%d, source=%d).", height(tLearn), height(sLearn));
end

for i = 1:height(tLearn)
    tVal = tLearn.Value{i};
    sVal = sLearn.Value{i};

    % Both should be dlarray/single
    tLearn.Value{i} = (1 - tau) .* tVal + tau .* sVal;
end

% Assign back (R2023b allows updating Value entries through Learnables table)
targetNet.Learnables = tLearn;
end
