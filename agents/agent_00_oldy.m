% function agent = agent_00_oldy(observationInfo, actionInfo)
% 
% numObs = prod(observationInfo.Dimension);
% numActions = numel(actionInfo.Elements);
% 
% criticNetwork = [
%     featureInputLayer(numObs, "Name", "observation", "Normalization","none")
%     fullyConnectedLayer(128, "Name", "fc1")
%     reluLayer("Name", "relu1")
%     fullyConnectedLayer(128, "Name", "fc2")
%     reluLayer("Name", "relu2")
%     fullyConnectedLayer(64, "Name", "fc3")
%     reluLayer("Name", "relu3")
%     fullyConnectedLayer(numActions, "Name", "output")
% ];
% 
% opt = rlRepresentationOptions( ...
%     'LearnRate', 1e-3, ...
%     'L2RegularizationFactor', 1e-5, ...
%     'Optimizer', 'adam', ...
%     'GradientThreshold', 1);
% 
% critic = rlQValueRepresentation( ...
%     criticNetwork, observationInfo, actionInfo, ...
%     'Observation', {'observation'}, opt);
% 
% agentOptions = rlDQNAgentOptions( ...
%     'UseDoubleDQN', true, ...
%     'SequenceLength', 1, ...
%     'TargetUpdateMethod','smoothing', ...
%     'TargetSmoothFactor', 1e-3, ...
%     'TargetUpdateFrequency', 1, ...
%     'ResetExperienceBufferBeforeTraining', true, ...
%     'SaveExperienceBufferWithAgent', true, ...
%     'MiniBatchSize', 64, ...
%     'NumStepsToLookAhead', 3, ...
%     'ExperienceBufferLength', 200000, ...
%     'DiscountFactor', 0.99);
% 
% agentOptions.EpsilonGreedyExploration.Epsilon = 1.0;
% agentOptions.EpsilonGreedyExploration.EpsilonMin = 0.02;
% agentOptions.EpsilonGreedyExploration.EpsilonDecay = 2e-4;
% 
% agent = rlDQNAgent(critic, agentOptions);
% 
% bufferLength = agentOptions.ExperienceBufferLength;
% perBuffer = rlPrioritizedReplayMemory(observationInfo, actionInfo, bufferLength);
% 
% perBuffer.PriorityExponent = 0.6;
% perBuffer.InitialImportanceSamplingExponent = 0.4;
% perBuffer.NumAnnealingSteps = 200000;
% 
% agent.ExperienceBuffer = perBuffer;
% 
% end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% 48 FEATURES               %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function agent = agent_00_oldy(observationInfo, actionInfo)
numObs = prod(observationInfo.Dimension);
numActions = numel(actionInfo.Elements);

criticNetwork = [
    featureInputLayer(numObs, "Name", "observation", "Normalization","none")
    fullyConnectedLayer(64, "Name", "fc_1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.1, "Name", "dropout1")
    fullyConnectedLayer(32, "Name", "fc_2")
    reluLayer("Name", "relu2")
    dropoutLayer(0.1, "Name", "dropout2")
    fullyConnectedLayer(32, "Name", "fc_3")
    reluLayer("Name", "relu3")
    fullyConnectedLayer(numActions, "Name", "output")
];

opt = rlRepresentationOptions( ...
    'LearnRate', 1e-4, ...
    'L2RegularizationFactor', 5e-5, ...
    'Optimizer', 'adam');

opt.OptimizerParameters.GradientDecayFactor = 0.85;
opt.OptimizerParameters.Momentum = 0.85;

critic = rlQValueRepresentation( ...
    criticNetwork, observationInfo, actionInfo, ...
    'Observation', {'observation'}, opt);

agentOptions = rlDQNAgentOptions( ...
    'UseDoubleDQN', true, ...
    'SequenceLength', 1, ...
    'TargetUpdateMethod','smoothing', ...
    'TargetSmoothFactor', 0.05, ...
    'TargetUpdateFrequency', 3, ...
    'ResetExperienceBufferBeforeTraining', false, ...
    'SaveExperienceBufferWithAgent', true, ...
    'MiniBatchSize', 64, ...
    'NumStepsToLookAhead', 2, ...
    'ExperienceBufferLength', 1e6, ...
    'DiscountFactor', 0.97);

agentOptions.EpsilonGreedyExploration.EpsilonDecay = 15e-5;
agentOptions.EpsilonGreedyExploration.Epsilon = 1;
agentOptions.EpsilonGreedyExploration.EpsilonMin = 0.001;

agent = rlDQNAgent(critic, agentOptions);

bufferLength = agentOptions.ExperienceBufferLength;
perBuffer = rlPrioritizedReplayMemory(observationInfo, actionInfo, bufferLength);

perBuffer.PriorityExponent = 0.6;
perBuffer.InitialImportanceSamplingExponent = 0.4;
perBuffer.NumAnnealingSteps = bufferLength;

agent.ExperienceBuffer = perBuffer;
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% 44 FEATURES               %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function agent = agent_00_oldy(observationInfo, actionInfo)
% % Configuración de Red Neuronal con Dropout para Regularización
% hL = @reluLayer;  % Activación tanh
% numActions = numel(actionInfo.Elements);
% disp(numActions)
% criticNetwork = [
%     featureInputLayer(44, "Name", "observation")  % Entrada con 44 características
%     fullyConnectedLayer(64, "Name", "fc_1")
%     hL("Name", "hL1")  % Tanh
%     dropoutLayer(0.1, "Name", "dropout1") % Apaga el 20% de las neuronas
%     fullyConnectedLayer(32, "Name", "fc_2")
%     hL("Name", "hL2")
%     dropoutLayer(0.1, "Name", "dropout2") 
%     fullyConnectedLayer(32, "Name", "fc_3")
%     hL("Name", "hL3")
%     fullyConnectedLayer(numActions, "Name", "output") % 81 acciones posibles
% ];
% 
% % Opciones del optimizador ajustadas
% opt = rlRepresentationOptions( ...
%     'LearnRate', 1e-4, ... % Reducimos la tasa de aprendizaje
%     'L2RegularizationFactor', 5e-5, ... % Regularización ligera
%     'Optimizer', 'adam' ...
% );
% 
% % Configuración del optimizador Adam
% opt.OptimizerParameters.GradientDecayFactor = 0.85;%0.95;
% opt.OptimizerParameters.Momentum = 0.85;%0.95;
% 
% critic = rlQValueRepresentation(criticNetwork, observationInfo, ...
%     actionInfo, 'Observation', {'observation'}, opt);
% 
% %% Opciones del Agente DQN
% agentOptions = rlDQNAgentOptions(...
%     'UseDoubleDQN', true, ...
%     'SequenceLength', 1, ...
%     'TargetUpdateMethod','smoothing', ...
%     'TargetSmoothFactor', 0.05, ...
%     'TargetUpdateFrequency', 3, ...
%     'ResetExperienceBufferBeforeTraining', false, ...
%     'SaveExperienceBufferWithAgent', true, ...
%     'MiniBatchSize', 64, ...
%     'NumStepsToLookAhead', 2, ...
%     'ExperienceBufferLength', 1e6, ...
%     'DiscountFactor', 0.97);%0.9995
% 
% % Exploración-Explotación ajustada
% agentOptions.EpsilonGreedyExploration.EpsilonDecay =15e-5;
% agentOptions.EpsilonGreedyExploration.Epsilon = 1;
% agentOptions.EpsilonGreedyExploration.EpsilonMin = 0.001;
% 
% agent = rlDQNAgent(critic, agentOptions);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%***PER***%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tamaño del buffer (ajusta si ya tienes uno en configs)
% agent = rlDQNAgent(critic, agentOptions);

% ====== PER (Prioritized Experience Replay) ======
% bufferLength = agentOptions.ExperienceBufferLength;
% 
% perBuffer = rlPrioritizedReplayMemory(observationInfo, actionInfo, bufferLength);
% 
% perBuffer.PriorityExponent = 0.6;
% perBuffer.InitialImportanceSamplingExponent = 0.4;
% perBuffer.NumAnnealingSteps = bufferLength;
% 
% agent.ExperienceBuffer = perBuffer;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

% hL = @(varargin) leakyReluLayer(0.01, varargin{:});  % Activación Leaky ReLU
% numActions = numel(actionInfo.Elements);
% 
% % Red neuronal para el crítico (Q-network)
% criticNetwork = [
%     featureInputLayer(44, "Name", "observation")
%     fullyConnectedLayer(64, "Name", "fc_1")
%     layerNormalizationLayer("Name", "norm1")
%     hL("Name", "hL1")
%     dropoutLayer(0.2, "Name", "dropout1")
% 
%     fullyConnectedLayer(32, "Name", "fc_2")
%     layerNormalizationLayer("Name", "norm2")
%     hL("Name", "hL2")
%     dropoutLayer(0.2, "Name", "dropout2")
% 
%     fullyConnectedLayer(32, "Name", "fc_3")
%     hL("Name", "hL3")
% 
%     fullyConnectedLayer(numActions, "Name", "output")
% ];
% 
% % Opciones del optimizador
% opt = rlRepresentationOptions( ...
%     'LearnRate', 1e-4, ...
%     'L2RegularizationFactor', 5e-4, ...
%     'Optimizer', 'adam');
% opt.OptimizerParameters.GradientDecayFactor = 0.95;
% opt.OptimizerParameters.Momentum = 0.9;
% 
% % Representación del crítico
% critic = rlQValueRepresentation(criticNetwork, observationInfo, ...
%     actionInfo, 'Observation', {'observation'}, opt);
% 
% % Opciones del agente DQN
% agentOptions = rlDQNAgentOptions(...
%     'UseDoubleDQN', true, ...
%     'SequenceLength', 1, ...
%     'TargetUpdateMethod', 'smoothing', ...
%     'TargetSmoothFactor', 0.05, ...
%     'TargetUpdateFrequency', 10, ...
%     'ResetExperienceBufferBeforeTraining', false, ...
%     'SaveExperienceBufferWithAgent', true, ...
%     'MiniBatchSize', 64, ... %cambiar a 16, 32, 64
%     'ExperienceBufferLength', 10000, ...
%     'NumStepsToLookAhead', 3, ...
%     'DiscountFactor', 0.995);
% 
% % Exploración-explotación
% agentOptions.EpsilonGreedyExploration.Epsilon = 1;
% agentOptions.EpsilonGreedyExploration.EpsilonMin = 0.01;
% agentOptions.EpsilonGreedyExploration.EpsilonDecay = 5e-4;
% 
% % Crear agente
% agent = rlDQNAgent(critic, agentOptions);
% end
