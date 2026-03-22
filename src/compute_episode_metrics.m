function metrics = compute_episode_metrics(qLog, qRefLog, aRawLog, aAppliedLog, period, thrSuccess)
% ============================================================
% Calcula métricas de tracking por episodio
%
% Entradas:
%   qLog       [T x M] estado real por paso
%   qRefLog    [T x M] referencia por paso
%   aRawLog    [T x M] acción discreta
%   aAppliedLog[T x M] acción aplicada
%   period     escalar, tiempo por step
%   thrSuccess umbral de éxito por motor
%
% Salida:
%   metrics    struct con métricas globales y por motor
% ============================================================

    if nargin < 6
        thrSuccess = 0.20;
    end

    % --------------------------------------------------------
    % 0) Limpiar NaNs
    % --------------------------------------------------------
    validRows = all(~isnan(qLog),2) & all(~isnan(qRefLog),2);
    qLog = qLog(validRows,:);
    qRefLog = qRefLog(validRows,:);

    if nargin >= 3 && ~isempty(aRawLog)
        aRawLog = aRawLog(validRows,:);
    else
        aRawLog = [];
    end

    if nargin >= 4 && ~isempty(aAppliedLog)
        aAppliedLog = aAppliedLog(validRows,:);
    else
        aAppliedLog = [];
    end

    T = size(qLog,1);
    M = size(qLog,2);

    if T == 0
        error('No hay datos válidos en qLog/qRefLog.');
    end

    e = qLog - qRefLog;
    absE = abs(e);
    sqE = e.^2;

    % --------------------------------------------------------
    % 1) Métricas globales básicas
    % --------------------------------------------------------
    metrics.MAE = mean(absE(:), 'omitnan');
    metrics.MSE = mean(sqE(:), 'omitnan');
    metrics.IAE = sum(absE(:), 'omitnan');
    metrics.ISE = sum(sqE(:), 'omitnan');

    % --------------------------------------------------------
    % 2) Métricas por motor
    % --------------------------------------------------------
    metrics.MAE_perMotor = mean(absE, 1, 'omitnan');
    metrics.MSE_perMotor = mean(sqE, 1, 'omitnan');
    metrics.IAE_perMotor = sum(absE, 1, 'omitnan');
    metrics.ISE_perMotor = sum(sqE, 1, 'omitnan');

    % --------------------------------------------------------
    % 3) Success global y por motor
    % --------------------------------------------------------
    successPerStepGlobal = all(absE < thrSuccess, 2);
    successPerStepMotor  = absE < thrSuccess;

    metrics.SuccessRate_Global = mean(successPerStepGlobal, 'omitnan');
    metrics.SuccessRate_perMotor = mean(successPerStepMotor, 1, 'omitnan');

    % --------------------------------------------------------
    % 4) Error estacionario
    %    promedio en los últimos N pasos
    % --------------------------------------------------------
    Nss = min(5, T);
    idxSS = (T-Nss+1):T;

    metrics.SteadyStateError_Global = mean(absE(idxSS,:), 'all', 'omitnan');
    metrics.SteadyStateError_perMotor = mean(absE(idxSS,:), 1, 'omitnan');

    % --------------------------------------------------------
    % 5) Overshoot por motor
    %    overshoot = max(q - qRef, 0)
    % --------------------------------------------------------
    overs = max(qLog - qRefLog, 0);
    metrics.Overshoot_perMotor = max(overs, [], 1, 'omitnan');
    metrics.Overshoot_Global = mean(metrics.Overshoot_perMotor, 'omitnan');

    % --------------------------------------------------------
    % 6) Correlación q vs q_ref por motor
    % --------------------------------------------------------
    corrVals = NaN(1,M);
    for m = 1:M
        q = qLog(:,m);
        qr = qRefLog(:,m);
        if std(q) > 0 && std(qr) > 0
            C = corrcoef(q, qr);
            corrVals(m) = C(1,2);
        end
    end
    metrics.Corr_perMotor = corrVals;
    metrics.Corr_Global = mean(corrVals, 'omitnan');

    % --------------------------------------------------------
    % 7) Delay aproximado por motor
    %    usando primer cambio apreciable
    % --------------------------------------------------------
    metrics.DelaySteps_perMotor = NaN(1,M);
    metrics.DelayTime_perMotor  = NaN(1,M);

    refChangeThr = 0.02;
    qChangeThr   = 0.02;

    for m = 1:M
        qr = qRefLog(:,m);
        q  = qLog(:,m);

        ref0 = qr(1);
        q0   = q(1);

        idxRef = find(abs(qr - ref0) > refChangeThr, 1, 'first');
        idxQ   = find(abs(q  - q0 ) > qChangeThr,   1, 'first');

        if ~isempty(idxRef) && ~isempty(idxQ)
            metrics.DelaySteps_perMotor(m) = idxQ - idxRef;
            metrics.DelayTime_perMotor(m)  = (idxQ - idxRef) * period;
        end
    end

    metrics.DelaySteps_Global = mean(metrics.DelaySteps_perMotor, 'omitnan');
    metrics.DelayTime_Global  = mean(metrics.DelayTime_perMotor, 'omitnan');

    % --------------------------------------------------------
    % 8) Rise time aproximado por motor
    %    tiempo para alcanzar 80% del cambio total de la referencia
    % --------------------------------------------------------
    metrics.RiseTimeSteps_perMotor = NaN(1,M);
    metrics.RiseTime_perMotor      = NaN(1,M);

    risePct = 0.8;

    for m = 1:M
        q  = qLog(:,m);
        qr = qRefLog(:,m);

        q0 = q(1);
        qr0 = qr(1);
        qrEnd = qr(end);

        target = q0 + risePct * (qrEnd - q0);

        idxRise = find((q - q0) .* sign(qrEnd - q0) >= abs(target - q0), 1, 'first');

        if ~isempty(idxRise)
            metrics.RiseTimeSteps_perMotor(m) = idxRise - 1;
            metrics.RiseTime_perMotor(m)      = (idxRise - 1) * period;
        end
    end

    metrics.RiseTime_Global = mean(metrics.RiseTime_perMotor, 'omitnan');

    % --------------------------------------------------------
    % 9) Effort de control
    % --------------------------------------------------------
    if ~isempty(aAppliedLog)
        metrics.ControlEffort_Global = mean(abs(aAppliedLog), 'all', 'omitnan');
        metrics.ControlEffort_perMotor = mean(abs(aAppliedLog), 1, 'omitnan');
        metrics.ControlEnergy_Global = sum(aAppliedLog.^2, 'all', 'omitnan');
        metrics.ControlEnergy_perMotor = sum(aAppliedLog.^2, 1, 'omitnan');
    else
        metrics.ControlEffort_Global = NaN;
        metrics.ControlEffort_perMotor = NaN(1,M);
        metrics.ControlEnergy_Global = NaN;
        metrics.ControlEnergy_perMotor = NaN(1,M);
    end

    % --------------------------------------------------------
    % 10) Suavidad del movimiento
    %     mean(abs(diff(q)))
    % --------------------------------------------------------
    dq = diff(qLog, 1, 1);
    metrics.Smoothness_Global = mean(abs(dq), 'all', 'omitnan');
    metrics.Smoothness_perMotor = mean(abs(dq), 1, 'omitnan');

    % --------------------------------------------------------
    % 11) Error final
    % --------------------------------------------------------
    metrics.FinalAbsError_perMotor = absE(end,:);
    metrics.FinalMeanAbsError = mean(absE(end,:), 'omitnan');
    metrics.FinalMaxAbsError  = max(absE(end,:), [], 'omitnan');

    % --------------------------------------------------------
    % 12) Score compuesto simple
    % --------------------------------------------------------
    maeScore = 1 / (1 + metrics.MAE);
    delayScore = 1 / (1 + max(metrics.DelayTime_Global,0));
    successScore = metrics.SuccessRate_Global;

    metrics.TQS = 0.4 * maeScore + 0.2 * delayScore + 0.4 * successScore;
end
