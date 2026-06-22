%% analisis_por_motor_encoderDiag.m
clear
clc
close all

%% Cargar encoderDiag
[file, path] = uigetfile("*.mat", "Selecciona encoder_diag_ep_XXXX.mat");

if isequal(file,0)
    error("No seleccionaste archivo.");
end

load(fullfile(path,file))

q     = encoderDiag.adjustEncLog;
q_ref = encoderDiag.flexConvertedLog;
dq    = encoderDiag.dqLog;

steps = encoderDiag.steps;

fprintf("\n=========================================\n");
fprintf("ANALISIS POR MOTOR / DOF\n");
fprintf("Archivo: %s\n", file);
fprintf("Steps: %d\n", steps);
fprintf("=========================================\n");

%% Seguridad
q     = q(1:steps,:);
q_ref = q_ref(1:steps,:);
dq    = dq(1:steps,:);

err = q_ref - q;
absErr = abs(err);

%% Métricas por motor
MAE = mean(absErr, 1, 'omitnan');
RMSE = sqrt(mean(err.^2, 1, 'omitnan'));
FinalAE = absErr(end,:);
MaxAE = max(absErr, [], 1);

Corr = NaN(1,4);
CorrDQdErr = NaN(1,4);
MeanDQ = mean(abs(dq), 1, 'omitnan');

for m = 1:4
    if std(q(:,m),'omitnan') > 1e-9 && std(q_ref(:,m),'omitnan') > 1e-9
        Corr(m) = corr(q(:,m), q_ref(:,m), 'Rows','complete');
    end

    errNormMotor = absErr(:,m);

    dErrMotor = [0; errNormMotor(1:end-1) - errNormMotor(2:end)];

    if std(abs(dq(:,m)),'omitnan') > 1e-9 && std(dErrMotor,'omitnan') > 1e-9
        CorrDQdErr(m) = corr(abs(dq(:,m)), dErrMotor, 'Rows','complete');
    end
end

%% Tabla resumen
T = table( ...
    (1:4)', ...
    MAE', ...
    RMSE', ...
    FinalAE', ...
    MaxAE', ...
    Corr', ...
    MeanDQ', ...
    CorrDQdErr', ...
    'VariableNames', { ...
        'Motor', ...
        'MAE', ...
        'RMSE', ...
        'FinalAbsError', ...
        'MaxAbsError', ...
        'Corr_q_qref', ...
        'MeanAbsDQ', ...
        'Corr_absDQ_dErr' ...
    });

fprintf("\n===== TABLA POR MOTOR =====\n");
disp(T)

%% Métricas globales
fprintf("\n===== RESUMEN GLOBAL =====\n");
fprintf("MAE global       = %.4f\n", mean(MAE,'omitnan'));
fprintf("RMSE global      = %.4f\n", mean(RMSE,'omitnan'));
fprintf("Final AE global  = %.4f\n", mean(FinalAE,'omitnan'));
fprintf("Mean |dq| global = %.4f\n", mean(MeanDQ,'omitnan'));

%% Gráficas q vs q_ref por motor
figure
for m = 1:4
    subplot(4,1,m)
    plot(q(:,m),'b','LineWidth',1.2)
    hold on
    plot(q_ref(:,m),'r','LineWidth',1.2)
    ylabel(sprintf('M%d',m))
    grid on

    if m == 1
        title('q vs q\_ref por motor')
        legend('q encoder','q\_ref')
    end
end
xlabel('Step')

%% Gráficas error absoluto por motor
figure
for m = 1:4
    subplot(4,1,m)
    plot(absErr(:,m),'LineWidth',1.2)
    ylabel(sprintf('|e%d|',m))
    grid on

    if m == 1
        title('Error absoluto por motor')
    end
end
xlabel('Step')

%% Acciones usadas
A = encoderDiag.aRawLog;

[Auniq,~,ic] = unique(A,'rows');
counts = accumarray(ic,1);

fprintf("\n===== ACCIONES ÚNICAS =====\n");
fprintf("Total acciones únicas: %d\n", size(Auniq,1));

ActionTable = array2table(Auniq, ...
    'VariableNames', {'M1','M2','M3','M4'});
ActionTable.Count = counts;

disp(ActionTable)

%% Guardar resultados junto al archivo
outFile = fullfile(path, replace(file, ".mat", "_analisis_por_motor.mat"));
save(outFile, "T", "ActionTable", "MAE", "RMSE", "FinalAE", "MaxAE", "Corr", "MeanDQ", "CorrDQdErr");

fprintf("\nAnálisis guardado en:\n%s\n", outFile);