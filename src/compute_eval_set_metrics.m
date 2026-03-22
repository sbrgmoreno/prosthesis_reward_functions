function [metricsTable, summary] = compute_eval_set_metrics(evalEpisodes, period, thrSuccess)
% ============================================================
% evalEpisodes: cell array donde cada celda tiene un struct:
%   .qLog
%   .qRefLog
%   .aRawLog
%   .aAppliedLog
% ============================================================

    if nargin < 3
        thrSuccess = 0.20;
    end

    nEps = numel(evalEpisodes);

    Episode = zeros(nEps,1);
    MAE = zeros(nEps,1);
    MSE = zeros(nEps,1);
    SuccessRate = zeros(nEps,1);
    SteadyStateError = zeros(nEps,1);
    Delay = zeros(nEps,1);
    RiseTime = zeros(nEps,1);
    Overshoot = zeros(nEps,1);
    Corr = zeros(nEps,1);
    ControlEffort = zeros(nEps,1);
    Smoothness = zeros(nEps,1);
    FinalMeanAbsError = zeros(nEps,1);
    FinalMaxAbsError = zeros(nEps,1);
    TQS = zeros(nEps,1);

    for k = 1:nEps
        ep = evalEpisodes{k};

        met = compute_episode_metrics( ...
            ep.qLog, ep.qRefLog, ep.aRawLog, ep.aAppliedLog, period, thrSuccess);

        Episode(k) = k;
        MAE(k) = met.MAE;
        MSE(k) = met.MSE;
        SuccessRate(k) = met.SuccessRate_Global;
        SteadyStateError(k) = met.SteadyStateError_Global;
        Delay(k) = met.DelayTime_Global;
        RiseTime(k) = met.RiseTime_Global;
        Overshoot(k) = met.Overshoot_Global;
        Corr(k) = met.Corr_Global;
        ControlEffort(k) = met.ControlEffort_Global;
        Smoothness(k) = met.Smoothness_Global;
        FinalMeanAbsError(k) = met.FinalMeanAbsError;
        FinalMaxAbsError(k) = met.FinalMaxAbsError;
        TQS(k) = met.TQS;
    end

    metricsTable = table( ...
        Episode, MAE, MSE, SuccessRate, SteadyStateError, Delay, RiseTime, ...
        Overshoot, Corr, ControlEffort, Smoothness, ...
        FinalMeanAbsError, FinalMaxAbsError, TQS);

    summary = struct();
    summary.MAE_mean = mean(MAE, 'omitnan');
    summary.MSE_mean = mean(MSE, 'omitnan');
    summary.SuccessRate_mean = mean(SuccessRate, 'omitnan');
    summary.SteadyStateError_mean = mean(SteadyStateError, 'omitnan');
    summary.Delay_mean = mean(Delay, 'omitnan');
    summary.RiseTime_mean = mean(RiseTime, 'omitnan');
    summary.Overshoot_mean = mean(Overshoot, 'omitnan');
    summary.Corr_mean = mean(Corr, 'omitnan');
    summary.ControlEffort_mean = mean(ControlEffort, 'omitnan');
    summary.Smoothness_mean = mean(Smoothness, 'omitnan');
    summary.FinalMeanAbsError_mean = mean(FinalMeanAbsError, 'omitnan');
    summary.FinalMaxAbsError_mean = mean(FinalMaxAbsError, 'omitnan');
    summary.TQS_mean = mean(TQS, 'omitnan');
end

