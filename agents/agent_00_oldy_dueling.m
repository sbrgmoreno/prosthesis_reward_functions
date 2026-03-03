function agent = agent_00_oldy_dueling(observationInfo, actionInfo)

hL = @reluLayer;
numActions = numel(actionInfo.Elements);
numObs = prod(observationInfo.Dimension);

% =========================
% 1) RED DUELING (SIN DROPOUT)
% =========================
% Tronco compartido
statePath = [
    featureInputLayer(numObs, "Name", "observation", "Normalization","none")
    fullyConnectedLayer(128, "Name", "fc1")
    hL("Name", "relu1")
    fullyConnectedLayer(128, "Name", "fc2")
    hL("Name", "relu2")
];

% Value head V(s)
valuePath = [
    fullyConnectedLayer(64, "Name", "v_fc1")
    hL("Name", "v_relu1")
    fullyConnectedLayer(1, "Name", "V")
];

% Advantage head A(s,a)
advPath = [
    fullyConnectedLayer(64, "Name", "a_fc1")
    hL("Name", "a_relu1")
    fullyConnectedLayer(numActions, "Name", "A")
];

% Grafo
lgraph = layerGraph(statePath);
lgraph = addLayers(lgraph, valuePath);
lgraph = addLayers(lgraph, advPath);

lgraph = connectLayers(lgraph, "relu2", "v_fc1");
lgraph = connectLayers(lgraph, "relu2", "a_fc1");

% Concat [V;A]
concatVA = concatenationLayer(1, 2, "Name", "concat_VA");
lgraph = addLayers(lgraph, concatVA);
lgraph = connectLayers(lgraph, "V", "concat_VA/in1");
lgraph = connectLayers(lgraph, "A", "concat_VA/in2");

% Dueling combine -> Q(s,a)
duel = DuelingCombineLayer(numActions, "dueling_combine");
lgraph = addLayers(lgraph, duel);
lgraph = connectLayers(lgraph, "concat_VA", "dueling_combine");

% =========================
% 2) CRITIC OPTIONS (ESTABLE)
% =========================
repOpts = rlRepresentationOptions( ...
    'Optimizer','adam', ...
    'LearnRate', 1e-3, ...                 % (1e-4 puede ser lento; 1e-3 suele ir mejor)
    'L2RegularizationFactor', 1e-5, ...
    'GradientThreshold', 1);               % anti-explosión

% Si quieres fijar betas de Adam (opcional; puedes comentar si no existe en tu versión)
% repOpts.OptimizerParameters.GradientDecayFactor = 0.9;
% repOpts.OptimizerParameters.SquaredGradientDecayFactor = 0.999;

critic = rlQValueRepresentation(lgraph, observationInfo, actionInfo, ...
    'Observation', {'observation'}, repOpts);

% =========================
% 3) DQN OPTIONS (DUELING + DOUBLE)
% =========================
agentOptions = rlDQNAgentOptions( ...
    'UseDoubleDQN', true, ...
    'MiniBatchSize', 64, ...
    'DiscountFactor', 0.99, ...
    'NumStepsToLookAhead', 3, ...          % más estable que 10
    'ExperienceBufferLength', 200000, ...  % en vez de 1e6
    'ResetExperienceBufferBeforeTraining', true, ...
    'SaveExperienceBufferWithAgent', true, ...
    'SequenceLength', 1);

% ---- Target update (elige UNA estrategia)

% (A) smoothing estable
agentOptions.TargetUpdateMethod   = 'smoothing';
agentOptions.TargetSmoothFactor   = 1e-3;
agentOptions.TargetUpdateFrequency = 1;

% % (B) periodic (alternativa)
% agentOptions.TargetUpdateMethod   = 'periodic';
% agentOptions.TargetUpdateFrequency = 200;

% Exploración (más sana)
agentOptions.EpsilonGreedyExploration.Epsilon     = 1.0;
agentOptions.EpsilonGreedyExploration.EpsilonMin  = 0.02;
agentOptions.EpsilonGreedyExploration.EpsilonDecay = 2e-4;

agent = rlDQNAgent(critic, agentOptions);

% =========================
% 4) PER (Prioritized Replay)
% =========================
bufferLength = agentOptions.ExperienceBufferLength;
perBuffer = rlPrioritizedReplayMemory(observationInfo, actionInfo, bufferLength);

perBuffer.PriorityExponent = 0.6;                 % alpha
perBuffer.InitialImportanceSamplingExponent = 0.4; % beta0
perBuffer.NumAnnealingSteps = 200000;             % no uses bufferLength si es enorme

agent.ExperienceBuffer = perBuffer;

end

