%% train_emg_to_qref_predictor.m
clear
clc
close all

fprintf('\n=========================================\n');
fprintf('TRAIN EMG -> q_ref PREDICTOR\n');
fprintf('=========================================\n');

%% 1) Cargar dataset supervisado

load("EMG_GLOVE_DATASET.mat")   % X, Y

fprintf('\nDataset cargado:\n');
fprintf('X size = %s\n', mat2str(size(X)));
fprintf('Y size = %s\n', mat2str(size(Y)));

%% 2) Normalizar glove a [0,1]

gloveScale = [4092 2046 1023 2046];

Ynorm = Y ./ gloveScale;
Ynorm = max(0, min(1, Ynorm));

%% 3) Limpieza

validRows = all(isfinite(X),2) & all(isfinite(Ynorm),2);

X = X(validRows,:);
Ynorm = Ynorm(validRows,:);

fprintf('\nDataset limpio:\n');
fprintf('X size = %s\n', mat2str(size(X)));
fprintf('Ynorm size = %s\n', mat2str(size(Ynorm)));

%% 4) Train/test split

rng(7)

N = size(X,1);
idx = randperm(N);

nTrain = round(0.8*N);

idxTrain = idx(1:nTrain);
idxTest  = idx(nTrain+1:end);

Xtrain = X(idxTrain,:);
Ytrain = Ynorm(idxTrain,:);

Xtest = X(idxTest,:);
Ytest = Ynorm(idxTest,:);

%% 5) Entrenar modelo ensemble, uno por motor/dedo

models = cell(1,4);
Ypred = zeros(size(Ytest));

fprintf('\nEntrenando Bagged Trees...\n');

for m = 1:4
    fprintf('Entrenando salida %d/4...\n', m);

    models{m} = fitrensemble( ...
        Xtrain, Ytrain(:,m), ...
        'Method','Bag', ...
        'NumLearningCycles',150, ...
        'Learners','tree' ...
    );

    Ypred(:,m) = predict(models{m}, Xtest);
end

Ypred = max(0, min(1, Ypred));

%% 6) Métricas

metrics = table();

for m = 1:4
    ytrue = Ytest(:,m);
    yhat  = Ypred(:,m);

    mae  = mean(abs(ytrue - yhat),'omitnan');
    rmse = sqrt(mean((ytrue - yhat).^2,'omitnan'));
    r2   = 1 - sum((ytrue - yhat).^2,'omitnan') / ...
               sum((ytrue - mean(ytrue,'omitnan')).^2,'omitnan');

    metrics.Finger(m,1) = m;
    metrics.MAE(m,1) = mae;
    metrics.RMSE(m,1) = rmse;
    metrics.R2(m,1) = r2;
end

fprintf('\n===== METRICAS TEST =====\n');
disp(metrics)

fprintf('\nPromedio MAE  = %.4f\n', mean(metrics.MAE));
fprintf('Promedio RMSE = %.4f\n', mean(metrics.RMSE));
fprintf('Promedio R2   = %.4f\n', mean(metrics.R2));

%% 7) Gráfica real vs predicho

figure
for m = 1:4
    subplot(2,2,m)
    scatter(Ytest(:,m), Ypred(:,m), 8, 'filled')
    hold on
    plot([0 1],[0 1],'k--','LineWidth',1.2)
    title(sprintf('q_ref predictor - DOF %d',m))
    xlabel('Real')
    ylabel('Predicho')
    grid on
end

%% 8) Guardar predictor

qrefPredictor.models = models;
qrefPredictor.gloveScale = gloveScale;
qrefPredictor.metrics = metrics;
qrefPredictor.modelType = "BaggedTrees";
qrefPredictor.inputDim = size(X,2);
qrefPredictor.outputDim = 4;

save("qrefPredictor_EMG_BaggedTrees.mat", "qrefPredictor");

fprintf('\nPredictor guardado en qrefPredictor_EMG_BaggedTrees.mat\n');