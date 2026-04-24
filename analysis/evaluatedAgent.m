% ================================================================
% evaluateAgent.m
% Evalúa un agente ya entrenado con métricas de sensibilidad
% Estado: [EMG(40), q(4), dq(4)] = 48 features
%
% USO:
%   1) Cambiar la ruta del agente en la línea de load()
%   2) Ejecutar este script desde la carpeta raíz del proyecto
% ================================================================
clear; clc;
clear classes;
clear functions;
rehash;
%% 0) Cargar agente entrenado
% ---- CAMBIAR ESTA RUTA al agente que quieras evaluar ----
agentFile = "C:\trainedAgentsProtesisNew\00_oldy\_\DDQN_81_Espacio_Acciones\Num_Steps_100\V25_5000_Reward_NO_q_ref\V25_5000_Episodes\Agent5000.mat";
% ---------------------------------------------------------
fprintf("Cargando agente: %s\n", agentFile);
loadedData = load(agentFile);
% Detectar automáticamente el nombre de la variable del agente
varNames = fieldnames(loadedData);
agentVarNames = {'agent', 'saved_agent', 'agentToSave', 'trainedAgent'};
agent = [];
for i = 1:length(agentVarNames)
   if ismember(agentVarNames{i}, varNames)
       agent = loadedData.(agentVarNames{i});
       fprintf("Variable del agente detectada: '%s'\n", agentVarNames{i});
       break;
   end
end
if isempty(agent)
   fprintf("Variables encontradas en el .mat: ");
   disp(varNames');
   error("No se encontró variable de agente. Edita el script manualmente.");
end
%% 1) Dataset
addpath(genpath('.\src'))
addpath(genpath('.\config'))
addpath(genpath('.\lib'))
addpath(genpath('.\agents'))
configs = configurables();
[emg, glove] = getDataset(configs.dataset, configs.dataset_folder);
%% 2) Carpeta de episodios
agent_dir = string(fullfile(getenv("USERPROFILE"), "Documents", "MATLAB", "prosthesis_episodes"));
if ~exist(agent_dir, "dir")
   mkdir(agent_dir);
end
%% 3) Crear entorno
env = Env(agent_dir, true, emg, glove);
%% 4) Crear opciones de simulación
simOpts = rlSimulationOptions( ...
   "MaxSteps", env.maxNumberStepsInEpisodes, ...
   "NumSimulations", 1);
%% 5) Simular
fprintf("\n===== SIMULANDO EPISODIO =====\n");
experience = sim(agent, env, simOpts);
%% 6) Ver acciones usadas
fprintf("\n===== ACCIONES ÚNICAS USADAS =====\n");
uniqueActions = unique(env.aRawLog(1:env.c,:), 'rows');
fprintf("Total acciones únicas: %d\n", size(uniqueActions,1));
disp(uniqueActions);
%% 7) Analizar sensibilidad
fprintf("\n===== ANÁLISIS DE SENSIBILIDAD =====\n");
analyze_sensitivity(env);


