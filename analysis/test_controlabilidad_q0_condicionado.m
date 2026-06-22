%% test_controlabilidad_q0_condicionado.m
clear
clc
close all

fprintf('\n=========================================\n');
fprintf('TEST CONTROLABILIDAD CON q0 CONDICIONADO\n');
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

numRepeats = 10;
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

rows = [];

for aIdx = 1:size(actionsToTest,1)

    action = actionsToTest(aIdx,:);

    fprintf('\n=========================================\n');
    fprintf('Accion: %s\n', mat2str(action));
    fprintf('=========================================\n');

    for r = 1:numRepeats

        obs0 = reset(env);

        % Estado 56:
        % EMG 1:40, q 41:44, q_ref_pred 45:48, err_pred 49:52, dq 53:56
        q0 = obs0(41:44)';

        for s = 1:numStepsAction
            [obs, reward, isDone, ~] = step(env, action);

            if isDone
                break
            end
        end

        q1 = obs(41:44)';
        dqTotal = q1 - q0;

        rows = [rows; ...
            aIdx, r, action, q0, q1, dqTotal, reward, double(isDone)];

        fprintf('rep %02d | q0=%s | dq=%s\n', ...
            r, mat2str(q0,3), mat2str(dqTotal,4));
    end
end

varNames = { ...
    'ActionID','Repeat', ...
    'A1','A2','A3','A4', ...
    'q01','q02','q03','q04', ...
    'q11','q12','q13','q14', ...
    'dQ1','dQ2','dQ3','dQ4', ...
    'Reward','IsDone'};

T = array2table(rows, 'VariableNames', varNames);

fprintf('\n===== TABLA COMPLETA =====\n');
disp(T)

%% Resumen promedio por acción
Summary = groupsummary(T, "ActionID", ["mean","std"], ...
    ["dQ1","dQ2","dQ3","dQ4","q01","q02","q03","q04"]);

fprintf('\n===== RESUMEN POR ACCION =====\n');
disp(Summary)

%% Motor dominante por acción
fprintf('\n=========================================\n');
fprintf('MOTOR DOMINANTE POR ACCION\n');
fprintf('=========================================\n');

for aIdx = 1:size(actionsToTest,1)

    action = actionsToTest(aIdx,:);
    idx = T.ActionID == aIdx;

    meanDQ = [
        mean(T.dQ1(idx),'omitnan'), ...
        mean(T.dQ2(idx),'omitnan'), ...
        mean(T.dQ3(idx),'omitnan'), ...
        mean(T.dQ4(idx),'omitnan')];

    stdDQ = [
        std(T.dQ1(idx),'omitnan'), ...
        std(T.dQ2(idx),'omitnan'), ...
        std(T.dQ3(idx),'omitnan'), ...
        std(T.dQ4(idx),'omitnan')];

    [maxEff, mDom] = max(abs(meanDQ));

    fprintf('\nAccion %s\n', mat2str(action));
    fprintf('  meanDQ = %s\n', mat2str(meanDQ,5));
    fprintf('  stdDQ  = %s\n', mat2str(stdDQ,5));
    fprintf('  dominante = M%d | efecto = %.6f\n', mDom, meanDQ(mDom));

    if maxEff < 1e-4
        fprintf('  Interpretacion: sin efecto claro\n');
    elseif meanDQ(mDom) > 0
        fprintf('  Interpretacion: aumenta principalmente M%d\n', mDom);
    else
        fprintf('  Interpretacion: disminuye principalmente M%d\n', mDom);
    end
end

%% Relación q0 -> dq por motor
fprintf('\n=========================================\n');
fprintf('CORRELACION q0 vs dQ\n');
fprintf('=========================================\n');

for m = 1:4
    q0col = sprintf('q0%d',m);
    dQcol = sprintf('dQ%d',m);

    c = corr(T.(q0col), T.(dQcol), 'Rows','complete');

    fprintf('Motor %d: corr(q0%d, dQ%d) = %.4f\n', m, m, m, c);
end

save("controlabilidad_q0_condicionado_results.mat", "T", "Summary");

fprintf('\nResultados guardados en controlabilidad_q0_condicionado_results.mat\n');