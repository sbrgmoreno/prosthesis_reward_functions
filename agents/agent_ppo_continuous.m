function agent = agent_ppo_continuous(observationInfo, actionInfo)

numObs = prod(observationInfo.Dimension);
numAct = prod(actionInfo.Dimension);

%% =========================
% Actor network
% =========================
% Una sola red con dos salidas:
% - media de la acción
% - desviación estándar de la acción

commonPath = [
    featureInputLayer(numObs, Name="observation", Normalization="none")
    fullyConnectedLayer(128, Name="actor_fc1")
    reluLayer(Name="actor_relu1")
    fullyConnectedLayer(128, Name="actor_fc2")
    reluLayer(Name="actor_relu2")
];

meanPath = [
    fullyConnectedLayer(numAct, Name="mean_fc")
    tanhLayer(Name="mean_tanh")
];

stdPath = [
    fullyConnectedLayer(numAct, Name="std_fc")
    softplusLayer(Name="std_softplus")
];

actorNet = layerGraph(commonPath);
actorNet = addLayers(actorNet, meanPath);
actorNet = addLayers(actorNet, stdPath);

actorNet = connectLayers(actorNet, "actor_relu2", "mean_fc");
actorNet = connectLayers(actorNet, "actor_relu2", "std_fc");

actor = rlContinuousGaussianActor( ...
    actorNet, ...
    observationInfo, ...
    actionInfo, ...
    ActionMeanOutputNames="mean_tanh", ...
    ActionStandardDeviationOutputNames="std_softplus", ...
    ObservationInputNames="observation");

%% =========================
% Critic network
% =========================
criticNet = [
    featureInputLayer(numObs, Name="observation", Normalization="none")
    fullyConnectedLayer(128, Name="critic_fc1")
    reluLayer(Name="critic_relu1")
    fullyConnectedLayer(128, Name="critic_fc2")
    reluLayer(Name="critic_relu2")
    fullyConnectedLayer(1, Name="critic_output")
];

critic = rlValueFunction(criticNet, observationInfo, ...
    ObservationInputNames="observation");

%% =========================
% PPO options
% =========================
agentOpts = rlPPOAgentOptions( ...
    ExperienceHorizon=256, ...
    ClipFactor=0.2, ...
    EntropyLossWeight=0.01, ...
    MiniBatchSize=64, ...
    NumEpoch=3, ...
    AdvantageEstimateMethod="gae", ...
    GAEFactor=0.95, ...
    DiscountFactor=0.99, ...
    SampleTime=1);

agent = rlPPOAgent(actor, critic, agentOpts);

end
