%% =========================================================
% plot_evaluation_metrics.m
% Visualization of 50-episode evaluation metrics
% ==========================================================

clear; clc; close all;

%% Load CSV
T = readtable('C:\trainedAgentsProtesisNew\00_oldy\_\DDQN_17_Action_Space_Hybrid\Num_Max_Steps_100\V32_3000_Episodes-20260405T021856Z-1-001\Eval_32_3000_Episodes_50\evaluation_metrics_table.csv');

%% Basic info
disp(T.Properties.VariableNames)

episodes = T.Episode;

%% =========================================================
% 1) Boxplots of core error metrics
% =========================================================
figure('Name','Error Metrics Distribution');
boxplot([T.MAE, T.MSE, T.FinalMeanAbsError, T.FinalMaxAbsError], ...
    'Labels', {'MAE','MSE','FinalMeanAE','FinalMaxAE'});
ylabel('Metric value');
title('Distribution of Error Metrics across 50 Episodes');
grid on;

%% =========================================================
% 2) Per-episode trend plots
% =========================================================
figure('Name','Per-Episode Metrics');

subplot(2,2,1)
plot(episodes, T.MAE, '-o')
xlabel('Episode')
ylabel('MAE')
title('MAE per Episode')
grid on

subplot(2,2,2)
plot(episodes, T.MSE, '-o')
xlabel('Episode')
ylabel('MSE')
title('MSE per Episode')
grid on

subplot(2,2,3)
plot(episodes, T.FinalMeanAbsError, '-o')
xlabel('Episode')
ylabel('Final Mean Abs Error')
title('Final Mean Error per Episode')
grid on

subplot(2,2,4)
plot(episodes, T.FinalMaxAbsError, '-o')
xlabel('Episode')
ylabel('Final Max Abs Error')
title('Final Max Error per Episode')
grid on

%% =========================================================
% 3) Control quality metrics
% =========================================================
figure('Name','Control Quality');

subplot(2,2,1)
plot(episodes, T.Corr, '-o')
xlabel('Episode')
ylabel('Correlation')
title('Tracking Correlation')
yline(0,'--')
grid on

subplot(2,2,2)
plot(episodes, T.ControlEffort, '-o')
xlabel('Episode')
ylabel('Control Effort')
title('Control Effort')
grid on

subplot(2,2,3)
plot(episodes, T.Smoothness, '-o')
xlabel('Episode')
ylabel('Smoothness')
title('Action Smoothness')
grid on

subplot(2,2,4)
plot(episodes, T.TQS, '-o')
xlabel('Episode')
ylabel('TQS')
title('Tracking Quality Score')
grid on

%% =========================================================
% 4) Histogram distributions
% =========================================================
figure('Name','Histograms');

subplot(2,2,1)
histogram(T.MAE,10)
title('Histogram of MAE')
grid on

subplot(2,2,2)
histogram(T.Corr,10)
title('Histogram of Correlation')
grid on

subplot(2,2,3)
histogram(T.FinalMeanAbsError,10)
title('Histogram of Final Mean Error')
grid on

subplot(2,2,4)
histogram(T.TQS,10)
title('Histogram of TQS')
grid on

%% =========================================================
% 5) Correlation scatter plots
% =========================================================
figure('Name','Metric Relationships');

subplot(1,2,1)
scatter(T.MAE, T.TQS, 50, 'filled')
xlabel('MAE')
ylabel('TQS')
title('MAE vs TQS')
grid on

subplot(1,2,2)
scatter(T.ControlEffort, T.FinalMeanAbsError, 50, 'filled')
xlabel('Control Effort')
ylabel('Final Mean Error')
title('Effort vs Final Error')
grid on