clear
clc
close all

fprintf('=========================================\n');
fprintf('ANALISIS DEL STEP DE MINIMO ERROR\n');
fprintf('=========================================\n');

metricsFile = ...
"C:\trainedAgentsProtesisNew\00_oldy\_\26-06-18 00 31 5\custom_metrics_V0.mat";

load(metricsFile)

nEpisodes = length(metrics.minErrNorm);

minStep = nan(nEpisodes,1);
minValue = nan(nEpisodes,1);

encoderFolder = ...
"C:\trainedAgentsProtesisNew\00_oldy\_\26-06-18 00 31 5\EvalEpisodes_full_81_26-06-18_00_31_07\encoder_diagnostics";

for ep = 1:nEpisodes

    fileName = fullfile(encoderFolder,...
        sprintf('encoder_diag_ep_%04d.mat',ep));

    if ~isfile(fileName)
        continue
    end

    load(fileName,"encoderDiag")

    errNorm = encoderDiag.errNormLog(:);

    [minValue(ep),idx] = min(errNorm);

    minStep(ep) = idx;

end

fprintf('\n===== ESTADISTICAS =====\n');

fprintf('Mean step min error = %.2f\n',mean(minStep,'omitnan'));
fprintf('Median step min error = %.2f\n',median(minStep,'omitnan'));

fprintf('Min step = %.0f\n',min(minStep));
fprintf('Max step = %.0f\n',max(minStep));

figure
histogram(minStep)
xlabel('Step donde ocurre el minimo error')
ylabel('Cantidad de episodios')
title('Distribucion del step de minimo error')

figure
scatter(minStep,minValue,'.')
xlabel('Step minimo error')
ylabel('Valor minimo de error')
title('Min error vs Step')

save('analisis_min_error_step.mat',...
    'minStep',...
    'minValue')