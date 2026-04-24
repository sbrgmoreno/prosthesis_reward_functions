%% =========================================================
% analyze_joint_logs.m
% Analisis por articulacion de episodios guardados en .mat
% =========================================================

clear; clc; close all;

%% ---------------------------------------------------------
% CONFIG
% ----------------------------------------------------------
episodeFolder = 'C:\trainedAgentsProtesisNew\00_oldy\_\DDQN_17_Action_Space_Hybrid\Num_Max_Steps_100\V30_3000_Episodes_Logs_Motor\EvalEpisodes_hybrid_17_26-04-22_20_04_41';  % <-- cambia si hace falta
thrStrict = 0.20;
thrNear   = 0.30;

files = dir(fullfile(episodeFolder, 'episode_*.mat'));

if isempty(files)
    error('No se encontraron archivos episode_*.mat en %s', episodeFolder);
end

numEpisodes = numel(files);
numJoints   = 4;
jointNames  = {'Motor1','Motor2','Motor3','Motor4'};

%% ---------------------------------------------------------
% BUFFERS
% ----------------------------------------------------------
finalErrPerJoint      = NaN(numEpisodes, numJoints);
meanErrPerJoint       = NaN(numEpisodes, numJoints);
maxErrPerJoint        = NaN(numEpisodes, numJoints);
stdErrPerJoint        = NaN(numEpisodes, numJoints);

meanAbsDQPerJoint     = NaN(numEpisodes, numJoints);
maxAbsDQPerJoint      = NaN(numEpisodes, numJoints);

meanDirAgreePerJoint  = NaN(numEpisodes, numJoints);

strictSuccessPerJoint = NaN(numEpisodes, numJoints);
nearSuccessPerJoint   = NaN(numEpisodes, numJoints);

worstJointIdx         = NaN(numEpisodes,1);
worstJointErr         = NaN(numEpisodes,1);

episodeNumber         = NaN(numEpisodes,1);
validStepsPerEpisode  = NaN(numEpisodes,1);
numUniqueActions      = NaN(numEpisodes,1);
rewardSumEpisode      = NaN(numEpisodes,1);

%% ---------------------------------------------------------
% LOOP DE EPISODIOS
% ----------------------------------------------------------
for i = 1:numEpisodes
    S = load(fullfile(files(i).folder, files(i).name));

    if ~isfield(S,'qLog') || ~isfield(S,'qRefLog')
        warning('Archivo %s no tiene qLog/qRefLog. Se omite.', files(i).name);
        continue;
    end

    q    = S.qLog;
    qRef = S.qRefLog;

    % ------------------------------------------------------
    % Detectar filas validas
    % ------------------------------------------------------
    validRows = all(~isnan(q),2) & all(~isnan(qRef),2);

    if ~any(validRows)
        warning('Archivo %s no tiene filas validas.', files(i).name);
        continue;
    end

    q    = q(validRows,:);
    qRef = qRef(validRows,:);

    T = size(q,1);
    validStepsPerEpisode(i) = T;

    % ------------------------------------------------------
    % Error por articulacion
    % ------------------------------------------------------
    err    = qRef - q;
    absErr = abs(err);

    finalErrPerJoint(i,:) = absErr(end,:);
    meanErrPerJoint(i,:)  = mean(absErr,1,'omitnan');
    maxErrPerJoint(i,:)   = max(absErr,[],1);
    stdErrPerJoint(i,:)   = std(absErr,0,1,'omitnan');

    % ------------------------------------------------------
    % Movimiento por articulacion
    % ------------------------------------------------------
    if isfield(S,'dqLog')
        dq = S.dqLog(validRows,:);
    else
        dq = [zeros(1,numJoints); diff(q,1,1)];
    end

    meanAbsDQPerJoint(i,:) = mean(abs(dq),1,'omitnan');
    maxAbsDQPerJoint(i,:)  = max(abs(dq),[],1);

    % ------------------------------------------------------
    % Acuerdo direccional por articulacion
    % ------------------------------------------------------
    if isfield(S,'dirAgreeLog')
        dirAgree = double(S.dirAgreeLog(validRows,:));
        meanDirAgreePerJoint(i,:) = mean(dirAgree,1,'omitnan');
    end

    % ------------------------------------------------------
    % Exito por articulacion
    % ------------------------------------------------------
    strictSuccessPerJoint(i,:) = double(finalErrPerJoint(i,:) < thrStrict);
    nearSuccessPerJoint(i,:)   = double(finalErrPerJoint(i,:) < thrNear);

    % ------------------------------------------------------
    % Peor articulacion del episodio
    % ------------------------------------------------------
    [worstJointErr(i), worstJointIdx(i)] = max(finalErrPerJoint(i,:));

    % ------------------------------------------------------
    % Metadatos
    % ------------------------------------------------------
    if isfield(S,'episodeNumber')
        episodeNumber(i) = S.episodeNumber;
    else
        episodeNumber(i) = i;
    end

    if isfield(S,'numUniqueActions')
        numUniqueActions(i) = S.numUniqueActions;
    elseif isfield(S,'actionList')
        numUniqueActions(i) = size(unique(S.actionList,'rows'),1);
    elseif isfield(S,'aRawLog')
        A = S.aRawLog(validRows,:);
        numUniqueActions(i) = size(unique(A,'rows'),1);
    end

    if isfield(S,'rewardSum')
        rewardSumEpisode(i) = S.rewardSum;
    elseif isfield(S,'rewardLog')
        rewardSumEpisode(i) = sum(S.rewardLog(validRows), 'omitnan');
    end
end

%% ---------------------------------------------------------
% TABLA RESUMEN POR ARTICULACION
% ----------------------------------------------------------
jointSummary = table;
jointSummary.Joint = jointNames';

jointSummary.FinalErr_Mean = mean(finalErrPerJoint,1,'omitnan')';
jointSummary.FinalErr_Std  = std(finalErrPerJoint,0,1,'omitnan')';
jointSummary.FinalErr_Min  = min(finalErrPerJoint,[],1)';
jointSummary.FinalErr_Max  = max(finalErrPerJoint,[],1)';

jointSummary.MeanErr_Mean  = mean(meanErrPerJoint,1,'omitnan')';
jointSummary.MeanErr_Std   = std(meanErrPerJoint,0,1,'omitnan')';

jointSummary.MaxErr_Mean   = mean(maxErrPerJoint,1,'omitnan')';
jointSummary.MaxErr_Std    = std(maxErrPerJoint,0,1,'omitnan')';

jointSummary.StdErr_Mean   = mean(stdErrPerJoint,1,'omitnan')';

jointSummary.MeanAbsDQ     = mean(meanAbsDQPerJoint,1,'omitnan')';
jointSummary.MaxAbsDQ      = mean(maxAbsDQPerJoint,1,'omitnan')';

jointSummary.DirAgree_Mean = mean(meanDirAgreePerJoint,1,'omitnan')';

jointSummary.StrictSuccessPct = 100 * mean(strictSuccessPerJoint,1,'omitnan')';
jointSummary.NearSuccessPct   = 100 * mean(nearSuccessPerJoint,1,'omitnan')';

disp('=== RESUMEN POR ARTICULACION ===')
disp(jointSummary)

writetable(jointSummary, fullfile(episodeFolder,'joint_summary.csv'));

%% ---------------------------------------------------------
% TABLA RESUMEN POR EPISODIO
% ----------------------------------------------------------
episodeSummary = table;
episodeSummary.Episode          = episodeNumber;
episodeSummary.ValidSteps       = validStepsPerEpisode;
episodeSummary.NumUniqueActions = numUniqueActions;
episodeSummary.RewardSum        = rewardSumEpisode;
episodeSummary.WorstJointIdx    = worstJointIdx;
episodeSummary.WorstJointName   = string(jointNames(worstJointIdx))';
episodeSummary.WorstJointErr    = worstJointErr;

disp('=== RESUMEN POR EPISODIO ===')
disp(episodeSummary(1:min(10,height(episodeSummary)),:))

writetable(episodeSummary, fullfile(episodeFolder,'episode_joint_summary.csv'));

%% ---------------------------------------------------------
% FIGURA 1: Boxplot error final por articulacion
% ----------------------------------------------------------
figure('Name','Final Error by Joint');
boxplot(finalErrPerJoint, 'Labels', jointNames);
ylabel('Final Absolute Error');
title('Distribucion del error final por articulacion');
grid on;

%% ---------------------------------------------------------
% FIGURA 2: Boxplot error medio por articulacion
% ----------------------------------------------------------
figure('Name','Mean Error by Joint');
boxplot(meanErrPerJoint, 'Labels', jointNames);
ylabel('Mean Absolute Error');
title('Distribucion del error medio por articulacion');
grid on;

%% ---------------------------------------------------------
% FIGURA 3: Barras del error final promedio
% ----------------------------------------------------------
figure('Name','Average Final Error by Joint');
bar(mean(finalErrPerJoint,1,'omitnan'));
set(gca,'XTickLabel',jointNames);
ylabel('Average Final Absolute Error');
title('Error final promedio por articulacion');
grid on;

%% ---------------------------------------------------------
% FIGURA 4: Barras del movimiento promedio |dq|
% ----------------------------------------------------------
figure('Name','Average |dq| by Joint');
bar(mean(meanAbsDQPerJoint,1,'omitnan'));
set(gca,'XTickLabel',jointNames);
ylabel('Mean |dq|');
title('Movimiento promedio por articulacion');
grid on;

%% ---------------------------------------------------------
% FIGURA 5: Acuerdo direccional promedio por articulacion
% ----------------------------------------------------------
figure('Name','Direction Agreement by Joint');
bar(mean(meanDirAgreePerJoint,1,'omitnan'));
set(gca,'XTickLabel',jointNames);
ylabel('Mean Direction Agreement');
title('Agreement promedio por articulacion');
grid on;

%% ---------------------------------------------------------
% FIGURA 6: Heatmap de error final (episodios x articulaciones)
% ----------------------------------------------------------
figure('Name','Final Error Heatmap');
imagesc(finalErrPerJoint);
colorbar;
xlabel('Articulacion');
ylabel('Episodio');
title('Heatmap del error final por articulacion');
set(gca,'XTick',1:numJoints,'XTickLabel',jointNames);

%% ---------------------------------------------------------
% FIGURA 7: Peor articulacion por episodio
% ----------------------------------------------------------
figure('Name','Worst Joint per Episode');
scatter(1:numEpisodes, worstJointIdx, 70, worstJointErr, 'filled');
colormap parula;
colorbar;
xlabel('Episodio');
ylabel('Indice de la peor articulacion');
title('Peor articulacion por episodio');
yticks(1:4);
yticklabels(jointNames);
grid on;

%% ---------------------------------------------------------
% FIGURA 8: % de episodios bajo umbral por articulacion
% ----------------------------------------------------------
figure('Name','Joint Success Rates');
bar([100*mean(strictSuccessPerJoint,1,'omitnan'); ...
     100*mean(nearSuccessPerJoint,1,'omitnan')]');
set(gca,'XTickLabel',jointNames);
legend({'Strict < 0.20','Near < 0.30'}, 'Location','best');
ylabel('% de episodios');
title('Porcentaje de episodios bajo umbral por articulacion');
ylim([0 100]);
grid on;

%% ---------------------------------------------------------
% GUARDAR TODO
% ----------------------------------------------------------
save(fullfile(episodeFolder,'joint_analysis_results.mat'), ...
    'finalErrPerJoint','meanErrPerJoint','maxErrPerJoint','stdErrPerJoint', ...
    'meanAbsDQPerJoint','maxAbsDQPerJoint','meanDirAgreePerJoint', ...
    'strictSuccessPerJoint','nearSuccessPerJoint', ...
    'worstJointIdx','worstJointErr', ...
    'episodeNumber','validStepsPerEpisode','numUniqueActions','rewardSumEpisode', ...
    'jointSummary','episodeSummary');

disp('Analisis por articulacion completado.')
disp(['Resultados guardados en: ' episodeFolder])