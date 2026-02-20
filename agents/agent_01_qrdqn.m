function net = agent_01_qrdqn(observationInfo, actionInfo, numQuantiles)
% agent_01_qrdqn: QR-DQN network (distributional critic)
% Output size = numActions * numQuantiles

numObs = prod(observationInfo.Dimension);
numActions = numel(actionInfo.Elements);

layers = [
    featureInputLayer(numObs, "Name","obs")
    fullyConnectedLayer(64, "Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(32, "Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(numActions*numQuantiles, "Name","quantiles")
];

net = dlnetwork(layerGraph(layers));
end

