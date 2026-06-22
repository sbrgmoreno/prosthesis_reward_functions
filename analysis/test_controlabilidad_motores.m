%% test_controlabilidad_motores.m
clear
clc
close all

fprintf('\n=========================================\n');
fprintf('TEST DE CONTROLABILIDAD POR MOTOR\n');
fprintf('=========================================\n');

%% Preparar entorno
addpath(genpath('.\src'))
addpath(genpath('.\config'))
addpath(genpath('.\lib'))
addpath(genpath('.\agents'))

configs = configurables();

if configs.usePrerecorded
    [emg, glove] = getDataset(configs.dataset, configs.dataset_folder);
    env = Env("", true, emg, glove);
else
    env = Env("");
end

%% Parámetros del test
numMotors = 4;
numRepeats = 3;

actionsToTest = [
     1  0  0  0
    -1  0  0  0
     0  1  0  0
     0 -1  0  0
     0  0  1  0
     0  0 -1  0
     0  0  0  1
     0  0  0 -1
];

results = [];

fprintf('\nAcciones a probar:\n');
disp(actionsToTest)

%% Reset inicial
reset(env);

%% Ejecutar pruebas
row = 0;

for aIdx = 1:size(actionsToTest,1)

    forcedAction = actionsToTest(aIdx,:);

    fprintf('\n=========================================\n');
    fprintf('Probando accion: %s\n', mat2str(forcedAction));
    fprintf('=========================================\n');

    for r = 1:numRepeats

        fprintf('\nRepeticion %d/%d\n', r, numRepeats);

        reset(env);

        % Leer q inicial
        if ~isempty(env.adjustEnc)
            q0 = env.adjustEnc(end,:);
        else
            motorData0 = env.prosthesis.read();

            encMin = [-10, 0, -1, 0];
            encMax = [175, 235, 485, 185];

            qMat0 = (motorData0 - encMin) ./ (encMax - encMin);
            qMat0 = max(0, min(1, qMat0));
            q0 = qMat0(end,:);
        end

        % Aplicar acción forzada una vez
        [~, reward, isDone, ~] = step(env, forcedAction);

        % Leer q final
        q1 = env.adjustEnc(end,:);

        dq = q1 - q0;

        row = row + 1;

        results(row).Action = forcedAction;
        results(row).Repeat = r;
        results(row).q0 = q0;
        results(row).q1 = q1;
        results(row).dq = dq;
        results(row).reward = reward;
        results(row).isDone = isDone;

        fprintf('q0 = %s\n', mat2str(q0,4));
        fprintf('q1 = %s\n', mat2str(q1,4));
        fprintf('dq = %s\n', mat2str(dq,4));
    end
end

%% Convertir a tabla resumen
fprintf('\n=========================================\n');
fprintf('RESUMEN POR ACCION\n');
fprintf('=========================================\n');

summaryRows = [];

for aIdx = 1:size(actionsToTest,1)

    forcedAction = actionsToTest(aIdx,:);

    idx = false(length(results),1);

    for k = 1:length(results)
        idx(k) = isequal(results(k).Action, forcedAction);
    end

    dqMat = vertcat(results(idx).dq);

    meanDQ = mean(dqMat,1,'omitnan');
    stdDQ  = std(dqMat,0,1,'omitnan');

    summaryRows = [summaryRows; forcedAction meanDQ stdDQ];
end

SummaryTable = array2table(summaryRows, ...
    'VariableNames', { ...
    'A1','A2','A3','A4', ...
    'MeanDQ1','MeanDQ2','MeanDQ3','MeanDQ4', ...
    'StdDQ1','StdDQ2','StdDQ3','StdDQ4'});

disp(SummaryTable)

%% Interpretación automática
fprintf('\n=========================================\n');
fprintf('INTERPRETACION POR MOTOR\n');
fprintf('=========================================\n');

for m = 1:numMotors

    actionPlus = zeros(1,4);
    actionMinus = zeros(1,4);

    actionPlus(m) = 1;
    actionMinus(m) = -1;

    idxPlus = all(actionsToTest == actionPlus,2);
    idxMinus = all(actionsToTest == actionMinus,2);

    dqPlus = SummaryTable{idxPlus, sprintf('MeanDQ%d',m)};
    dqMinus = SummaryTable{idxMinus, sprintf('MeanDQ%d',m)};

    fprintf('\nMotor %d:\n', m);
    fprintf('  accion +1 -> MeanDQ%d = %.6f\n', m, dqPlus);
    fprintf('  accion -1 -> MeanDQ%d = %.6f\n', m, dqMinus);

    if dqPlus > 0 && dqMinus < 0
        fprintf('  Interpretacion: signo NORMAL\n');
    elseif dqPlus < 0 && dqMinus > 0
        fprintf('  Interpretacion: signo INVERTIDO\n');
    elseif abs(dqPlus) < 1e-4 && abs(dqMinus) < 1e-4
        fprintf('  Interpretacion: SIN RESPUESTA / DEAD-ZONE\n');
    else
        fprintf('  Interpretacion: RESPUESTA AMBIGUA\n');
    end
end

%% Guardar
save("controlabilidad_motores_results.mat", "results", "SummaryTable");

fprintf('\nResultados guardados en controlabilidad_motores_results.mat\n');