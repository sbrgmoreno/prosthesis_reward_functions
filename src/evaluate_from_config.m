clear; clc;

% ============================================================
% 1) Leer configuraciones
% ============================================================
params = configurables();

assert(params.run_training == false, ...
    'configurables.m debe estar en modo evaluacion: params.run_training = false');

numEpisodes = 50;   % puedes cambiarlo si quieres

% ============================================================
% 2) Cargar dataset si aplica
% ============================================================
% Ajusta esta parte segun tu flujo real de carga de datos.
% La idea es obtener emgs y gloveDatas para crear el entorno.

[emgs, gloveDatas] = loadDataset(params.dataset, params.dataset_folder);

% ============================================================
% 3) Crear entorno
% ============================================================
env = Env("", params.usePrerecorded, emgs, gloveDatas);

% ============================================================
% 4) Cargar agente entrenado
% ============================================================
S = load(params.agentFile);

% Intentar detectar el nombre del agente dentro del .mat
agent = [];
fn = fieldnames(S);
for i = 1:numel(fn)
    candidate = S.(fn{i});
    if isa(candidate, 'rl.agent.AbstractAgent') || contains(class(candidate), 'rl')
        agent = candidate;
        fprintf('Agente cargado desde variable: %s\n', fn{i});
        break;
    end
end

if isempty(agent)
    error('No se encontro un agente RL dentro del archivo .mat');
end

% ============================================================
% 5) Opciones de simulacion
% ============================================================
simOpts = params.simOpts;
simOpts.MaxSteps = params.maxNumberStepsInEpisodes;

% ============================================================
% 6) Preparar almacenamiento de episodios
% ============================================================
evalEpisodes = cell(numEpisodes,1);

% ============================================================
% 7) Evaluar episodio por episodio
% ============================================================
for k = 1:numEpisodes

    fprintf('\n=====================================\n');
    fprintf('Evaluando episodio %d de %d\n', k, numEpisodes);
    fprintf('=====================================\n');

    sim(agent, env, simOpts);

    ep = struct();
    ep.qLog        = env.qLog(1:env.c,:);
    ep.qRefLog     = env.qRefLog(1:env.c,:);
    ep.aRawLog     = env.aRawLog(1:env.c,:);
    ep.aAppliedLog = env.aAppliedLog(1:env.c,:);

    % opcionales
    ep.meanDistLog    = env.meanDistLog(1:env.c);
    ep.mseLog         = env.mseLog(1:env.c);
    ep.successLog     = env.successLog(1:env.c);
    ep.nearSuccessLog = env.nearSuccessLog(1:env.c);

    evalEpisodes{k} = ep;

    fprintf('  Steps guardados: %d\n', env.c);
    fprintf('  qLog size: [%d x %d]\n', size(ep.qLog,1), size(ep.qLog,2));
end

fprintf('\nTotal de episodios guardados: %d\n', numel(evalEpisodes));

% ============================================================
% 8) Calcular metricas
% ============================================================
[metricsTable, summary] = compute_eval_set_metrics(evalEpisodes, env.period, 0.20);

disp('===== TABLA POR EPISODIO =====');
disp(metricsTable);

disp('===== RESUMEN GLOBAL =====');
disp(summary);

% ============================================================
% 9) Guardar resultados
% ============================================================
writetable(metricsTable, 'evaluation_metrics_table.csv');

summaryTable = table( ...
    summary.MAE_mean, ...
    summary.MSE_mean, ...
    summary.SuccessRate_mean, ...
    summary.SteadyStateError_mean, ...
    summary.Delay_mean, ...
    summary.RiseTime_mean, ...
    summary.Overshoot_mean, ...
    summary.Corr_mean, ...
    summary.ControlEffort_mean, ...
    summary.Smoothness_mean, ...
    summary.FinalMeanAbsError_mean, ...
    summary.FinalMaxAbsError_mean, ...
    summary.TQS_mean, ...
    'VariableNames', { ...
        'MAE', 'MSE', 'SuccessRate', 'SteadyStateError', ...
        'Delay', 'RiseTime', 'Overshoot', 'Correlation', ...
        'ControlEffort', 'Smoothness', ...
        'FinalMeanAbsError', 'FinalMaxAbsError', 'TQS' ...
    });

disp('===== TABLA RESUMEN =====');
disp(summaryTable);

writetable(summaryTable, 'evaluation_summary_table.csv');
save('evaluation_evalEpisodes.mat', 'evalEpisodes', 'summary', 'summaryTable');