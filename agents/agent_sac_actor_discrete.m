function actorNet = agent_sac_actor_discrete(observationInfo, actionInfo)
% Discrete SAC Actor network
% Outputs logits for each discrete action.

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
    fullyConnectedLayer(numActions, "Name","logits")
];

actorNet = dlnetwork(layerGraph(layers));
end
