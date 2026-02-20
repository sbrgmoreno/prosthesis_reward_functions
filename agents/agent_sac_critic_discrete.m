function criticNet = agent_sac_critic_discrete(observationInfo, actionInfo)
% Discrete SAC Critic network
% Outputs Q-values for all discrete actions given a state.

numObs = prod(observationInfo.Dimension);
numActions = numel(actionInfo.Elements);

layers = [
    featureInputLayer(numObs, "Name","obs")
    fullyConnectedLayer(64, "Name","fc1")
    reluLayer("Name","relu1")
    dropoutLayer(0.1, "Name","drop1")
    fullyConnectedLayer(32, "Name","fc2")
    reluLayer("Name","relu2")
    dropoutLayer(0.1, "Name","drop2")
    fullyConnectedLayer(numActions, "Name","q_values")
];

criticNet = dlnetwork(layerGraph(layers));
end


