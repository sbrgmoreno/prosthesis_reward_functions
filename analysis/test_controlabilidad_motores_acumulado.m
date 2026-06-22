%% test_controlabilidad_motores_acumulado.m
clear
clc
close all

fprintf('\n=========================================\n');
fprintf('TEST ACUMULADO DE CONTROLABILIDAD POR MOTOR\n');
fprintf('=========================================\n');

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

%% Parámetros
numRepeats = 3;
numStepsAction = 5;

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

row = 0;

for aIdx = 1:size(actionsToTest,1)

    forcedAction = actionsToTest(aIdx,:);

    fprintf('\n=========================================\n');
    fprintf('Probando accion acumulada: %s\n', mat2str(forcedAction));
    fprintf('=========================================\n');

    for r = 1:numRepeats

        fprintf('\nRepeticion %d/%d\n', r, numRepeats);

        obs0 = reset(env);

        q0 = obs0(41:44)';

        for s = 1:numStepsAction
            [obs, reward, isDone, ~] = step(env, forcedAction);

            if isDone
                break
            end
        end

        q1 = obs(41:44)';
        dqTotal = q1 - q0;

        row = row + 1;

        results(row).Action = forcedAction;
        results(row).Repeat = r;
        results(row).NumSteps = s;
        results(row).q0 = q0;
        results(row).q1 = q1;
        results(row).dqTotal = dqTotal;
        results(row).reward = reward;
        results(row).isDone = isDone;

        fprintf('q0      = %s\n', mat2str(q0,4));
        fprintf('q1      = %s\n', mat2str(q1,4));
        fprintf('dqTotal = %s\n', mat2str(dqTotal,4));
    end
end

%% Resumen
fprintf('\n=========================================\n');
fprintf('RESUMEN ACUMULADO POR ACCION\n');
fprintf('=========================================\n');

summaryRows = [];

for aIdx = 1:size(actionsToTest,1)

    forcedAction = actionsToTest(aIdx,:);

    idx = false(length(results),1);

    for k = 1:length(results)
        idx(k) = isequal(results(k).Action, forcedAction);
    end

    dqMat = vertcat(results(idx).dqTotal);

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

%% Interpretación: qué encoder responde más a cada acción
fprintf('\n=========================================\n');
fprintf('MOTOR DOMINANTE POR ACCION\n');
fprintf('=========================================\n');

for aIdx = 1:size(actionsToTest,1)

    forcedAction = actionsToTest(aIdx,:);

    meanDQ = SummaryTable{aIdx, {'MeanDQ1','MeanDQ2','MeanDQ3','MeanDQ4'}};

    [maxEffect, idxMotor] = max(abs(meanDQ));

    fprintf('\nAccion %s\n', mat2str(forcedAction));
    fprintf('  MeanDQ = %s\n', mat2str(meanDQ,5));
    fprintf('  Motor/encoder dominante: M%d | efecto = %.6f\n', idxMotor, meanDQ(idxMotor));

    if maxEffect < 1e-4
        fprintf('  Interpretacion: sin respuesta clara\n');
    elseif meanDQ(idxMotor) > 0
        fprintf('  Interpretacion: aumenta encoder M%d\n', idxMotor);
    else
        fprintf('  Interpretacion: disminuye encoder M%d\n', idxMotor);
    end
end

%% Interpretación por motor nominal
fprintf('\n=========================================\n');
fprintf('INTERPRETACION POR MOTOR NOMINAL\n');
fprintf('=========================================\n');

for m = 1:4

    actionPlus = zeros(1,4);
    actionMinus = zeros(1,4);

    actionPlus(m) = 1;
    actionMinus(m) = -1;

    idxPlus = all(actionsToTest == actionPlus,2);
    idxMinus = all(actionsToTest == actionMinus,2);

    dqPlus = SummaryTable{idxPlus, sprintf('MeanDQ%d',m)};
    dqMinus = SummaryTable{idxMinus, sprintf('MeanDQ%d',m)};

    fprintf('\nMotor nominal %d:\n', m);
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
save("controlabilidad_motores_acumulado_results.mat", ...
    "results", "SummaryTable");

fprintf('\nResultados guardados en controlabilidad_motores_acumulado_results.mat\n');