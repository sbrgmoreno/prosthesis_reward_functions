%% analisis_supervisado_emg_glove.m

clear
clc
close all

fprintf('\n=========================================\n');
fprintf('ANALISIS SUPERVISADO EMG -> GLOVE\n');
fprintf('=========================================\n');

%% Cargar dataset alineado

load("EMG_GLOVE_DATASET.mat")   % X, Y

fprintf('\n===== DATASET ORIGINAL =====\n');
fprintf('X size = %s\n', mat2str(size(X)));
fprintf('Y size = %s\n', mat2str(size(Y)));

%% Normalizar glove

gloveScale = [4092 2046 1023 2046];

Ynorm = Y ./ gloveScale;
Ynorm = max(0, min(1, Ynorm));

fprintf('\n===== RANGOS Y NORMALIZACION =====\n');
disp('min(Y):')
disp(min(Y))

disp('max(Y):')
disp(max(Y))

disp('min(Ynorm):')
disp(min(Ynorm))

disp('max(Ynorm):')
disp(max(Ynorm))

%% Limpieza básica

validRows = all(isfinite(X),2) & all(isfinite(Ynorm),2);

X = X(validRows,:);
Ynorm = Ynorm(validRows,:);

fprintf('\n===== DATASET LIMPIO =====\n');
fprintf('X size = %s\n', mat2str(size(X)));
fprintf('Ynorm size = %s\n', mat2str(size(Ynorm)));

%% Partición train/test

rng(7)

N = size(X,1);
idx = randperm(N);

nTrain = round(0.8*N);

idxTrain = idx(1:nTrain);
idxTest  = idx(nTrain+1:end);

Xtrain = X(idxTrain,:);
Xtest  = X(idxTest,:);

Ytrain = Ynorm(idxTrain,:);
Ytest  = Ynorm(idxTest,:);

fprintf('\nTrain samples: %d\n', size(Xtrain,1));
fprintf('Test samples : %d\n', size(Xtest,1));

%% Funciones de métricas

computeMetrics = @(ytrue, ypred) struct( ...
    'MAE',  mean(abs(ytrue - ypred), 'omitnan'), ...
    'RMSE', sqrt(mean((ytrue - ypred).^2, 'omitnan')), ...
    'R2',   1 - sum((ytrue - ypred).^2, 'omitnan') ./ ...
              sum((ytrue - mean(ytrue,'omitnan')).^2, 'omitnan') ...
);

%% =========================================================
% MODELO 1: Regresión lineal
%% =========================================================

fprintf('\n=========================================\n');
fprintf('MODELO 1: FITRLINEAR\n');
fprintf('=========================================\n');

YpredLinear = zeros(size(Ytest));

modelsLinear = cell(1,4);

for m = 1:4
    fprintf('Entrenando dedo %d...\n', m);

    modelsLinear{m} = fitrlinear( ...
        Xtrain, Ytrain(:,m), ...
        'Learner','leastsquares', ...
        'Regularization','ridge', ...
        'Lambda',1e-3 ...
    );

    YpredLinear(:,m) = predict(modelsLinear{m}, Xtest);
end

YpredLinear = max(0, min(1, YpredLinear));

metricsLinear = table();

for m = 1:4
    met = computeMetrics(Ytest(:,m), YpredLinear(:,m));

    metricsLinear.Finger(m,1) = m;
    metricsLinear.MAE(m,1)    = met.MAE;
    metricsLinear.RMSE(m,1)   = met.RMSE;
    metricsLinear.R2(m,1)     = met.R2;
end

disp(metricsLinear)

%% =========================================================
% MODELO 2: Ensemble / Bagged Trees
%% =========================================================

fprintf('\n=========================================\n');
fprintf('MODELO 2: FITRENSEMBLE BAGGED TREES\n');
fprintf('=========================================\n');

YpredEns = zeros(size(Ytest));

modelsEns = cell(1,4);

for m = 1:4
    fprintf('Entrenando dedo %d...\n', m);

    modelsEns{m} = fitrensemble( ...
        Xtrain, Ytrain(:,m), ...
        'Method','Bag', ...
        'NumLearningCycles',100, ...
        'Learners','tree' ...
    );

    YpredEns(:,m) = predict(modelsEns{m}, Xtest);
end

YpredEns = max(0, min(1, YpredEns));

metricsEns = table();

for m = 1:4
    met = computeMetrics(Ytest(:,m), YpredEns(:,m));

    metricsEns.Finger(m,1) = m;
    metricsEns.MAE(m,1)    = met.MAE;
    metricsEns.RMSE(m,1)   = met.RMSE;
    metricsEns.R2(m,1)     = met.R2;
end

disp(metricsEns)

%% =========================================================
% Comparación gráfica real vs predicho
%% =========================================================

figure
for m = 1:4
    subplot(2,2,m)
    scatter(Ytest(:,m), YpredLinear(:,m), 8, 'filled')
    hold on
    plot([0 1],[0 1],'k--','LineWidth',1.2)
    title(sprintf('Linear - Finger %d',m))
    xlabel('Real')
    ylabel('Predicho')
    grid on
end

figure
for m = 1:4
    subplot(2,2,m)
    scatter(Ytest(:,m), YpredEns(:,m), 8, 'filled')
    hold on
    plot([0 1],[0 1],'k--','LineWidth',1.2)
    title(sprintf('Ensemble - Finger %d',m))
    xlabel('Real')
    ylabel('Predicho')
    grid on
end

%% Guardar resultados

results.metricsLinear = metricsLinear;
results.metricsEns = metricsEns;
results.modelsLinear = modelsLinear;
results.modelsEns = modelsEns;
results.gloveScale = gloveScale;

save("EMG_GLOVE_SUPERVISED_RESULTS.mat", "results");

fprintf('\nResultados guardados en EMG_GLOVE_SUPERVISED_RESULTS.mat\n');