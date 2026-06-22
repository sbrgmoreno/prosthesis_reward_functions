function q_ref_pred = predictQrefFromEMG(emgFeatures)
% predictQrefFromEMG predicts q_ref from 40 EMG features.
%
% Input:
%   emgFeatures : 40x1 or 1x40 EMG feature vector
%
% Output:
%   q_ref_pred  : 4x1 predicted normalized reference in [0,1]

    persistent qrefPredictorLoaded

    if isempty(qrefPredictorLoaded)
        S = load("qrefPredictor_EMG_BaggedTrees.mat", "qrefPredictor");
        qrefPredictorLoaded = S.qrefPredictor;
    end

    x = emgFeatures(:)';

    if size(x,2) ~= qrefPredictorLoaded.inputDim
        error("predictQrefFromEMG:InvalidInputSize", ...
            "Expected %d EMG features, got %d.", ...
            qrefPredictorLoaded.inputDim, size(x,2));
    end

    q_ref_pred = zeros(1, qrefPredictorLoaded.outputDim);

    for m = 1:qrefPredictorLoaded.outputDim
        q_ref_pred(m) = predict(qrefPredictorLoaded.models{m}, x);
    end

    q_ref_pred = max(0, min(1, q_ref_pred));
    q_ref_pred = q_ref_pred(:);
end