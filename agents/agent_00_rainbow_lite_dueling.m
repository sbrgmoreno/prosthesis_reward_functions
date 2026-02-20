function agent = agent_00_rainbow_lite_dueling(observationInfo, actionInfo)

hL = @reluLayer;
numActions = numel(actionInfo.Elements);

% Dimensión de entrada (en tu red actual es 44)
numObs = prod(observationInfo.Dimension);

% ---------- TRONCO compartido ----------
statePath = [
    featureInputLayer(numObs, "Name", "observation")
    fullyConnectedLayer(64, "Name", "fc_1")
    hL("Name", "hL1")
    dropoutLayer(0.1, "Name", "dropout1")
    fullyConnectedLayer(32, "Name", "fc_2")
    hL("Name", "hL2")
    dropoutLayer(0.1, "Name", "dropout2")
    fullyConnectedLayer(32, "Name", "fc_3")
    hL("Name", "hL3")
];

% ---------- VALUE head: V(s) -> 1 ----------
valuePath = [
    fullyConnectedLayer(32, "Name", "v_fc1")
    hL("Name", "v_h1")
    fullyConnectedLayer(1, "Name", "V")
];

% ---------- ADVANTAGE head: A(s,a) -> numActions ----------
advPath = [
    fullyConnectedLayer(32, "Name", "a_fc1")
    hL("Name", "a_h1")
    fullyConnectedLayer(numActions, "Name", "A")
];

% ---------- Construir grafo ----------
lgraph = layerGraph(statePath);
lgraph = addLayers(lgraph, valuePath);
lgraph = addLayers(lgraph, advPath);

lgraph = connectLayers(lgraph, "hL3", "v_fc1");
lgraph = connectLayers(lgraph, "hL3", "a_fc1");

% concat [V; A]
concatVA = concatenationLayer(1, 2, "Name", "concat_VA");
lgraph = addLayers(lgraph, concatVA);
lgraph = connectLayers(lgraph, "V", "concat_VA/in1");
lgraph = connectLayers(lgraph, "A", "concat_VA/in2");

% Dueling combine -> Q(s,a)
duel = DuelingCombineLayer(numActions, "dueling_combine");
lgraph = addLayers(lgraph, duel);
lgraph = connectLayers(lgraph, "concat_VA", "dueling_combine");

% ---------- Opciones de representación (igual que tu oldy) ----------
opt = rlRepresentationOptions( ...
    'LearnRate', 1e-4, ...
    'L2RegularizationFactor', 5e-5, ...
    'Optimizer', 'adam');

opt.OptimizerParameters.GradientDecayFactor = 0.85;
opt.OptimizerParameters.Momentum = 0.85;

% Critic con layerGraph (dueling)
critic = rlQValueRepresentation(lgraph, observationInfo, actionInfo, ...
    'Observation', {'observation'}, opt);

% ---------- Opciones DQN (igual que tu oldy + PER) ----------
agentOptions = rlDQNAgentOptions(...
    'UseDoubleDQN', true, ...
    'SequenceLength', 1, ...
    'TargetUpdateMethod','smoothing', ...
    'TargetSmoothFactor', 0.05, ...
    'TargetUpdateFrequency', 3, ...
    'ResetExperienceBufferBeforeTraining', false, ...
    'SaveExperienceBufferWithAgent', true, ...
    'MiniBatchSize', 64, ...
    'NumStepsToLookAhead', 3, ...
    'ExperienceBufferLength', 1e6, ...
    'DiscountFactor', 0.97);

agentOptions.EpsilonGreedyExploration.EpsilonDecay = 15e-5;
agentOptions.EpsilonGreedyExploration.Epsilon = 1;
agentOptions.EpsilonGreedyExploration.EpsilonMin = 0.001;

agent = rlDQNAgent(critic, agentOptions);

% ---------- PER ----------
bufferLength = agentOptions.ExperienceBufferLength;
perBuffer = rlPrioritizedReplayMemory(observationInfo, actionInfo, bufferLength);
perBuffer.PriorityExponent = 0.6;
perBuffer.InitialImportanceSamplingExponent = 0.4;
perBuffer.NumAnnealingSteps = 250000;
agent.ExperienceBuffer = perBuffer;

end

