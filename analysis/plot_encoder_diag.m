clear; clc; close all;

[file, path] = uigetfile('*.mat','Selecciona encoder_diag_ep_XXXX.mat');
S = load(fullfile(path,file));

D = S.encoderDiag;

q     = D.adjustEncLog;
q_ref = D.flexConvertedLog;
enc   = D.encRawLog;

N = min(size(q,1), size(q_ref,1));
q = q(1:N,:);
q_ref = q_ref(1:N,:);
enc = enc(1:N,:);
t = 1:N;


fprintf('\n===== OFFSET INICIAL q(1,:) - q_ref(1,:) =====\n');
initialOffset = q(1,:) - q_ref(1,:);
disp(initialOffset);

%% ===== Alineación por offset inicial =====
q_aligned     = q - q(1,:);
q_ref_aligned = q_ref - q_ref(1,:);


figure('Name','q vs q_ref alineados por offset inicial');

for i = 1:4
    subplot(4,1,i)

    plot(t, q_aligned(:,i), 'b', 'LineWidth', 1.5);
    hold on
    plot(t, q_ref_aligned(:,i), 'r', 'LineWidth', 1.5);

    grid on
    ylabel(['M',num2str(i)])
    title(['Motor ',num2str(i),' - q vs q\_ref alineados'])

    legend('q alineado','q\_ref alineado')
end

xlabel('Step')


figure('Name','q vs q_ref normalizados');
for i = 1:4
    subplot(4,1,i)
    plot(t,q(:,i),'b','LineWidth',1.5); hold on;
    plot(t,q_ref(:,i),'r','LineWidth',1.5);
    grid on;
    ylabel(['M',num2str(i)]);
    title(['Motor ',num2str(i),' q vs q\_ref']);
    legend('q encoder','q\_ref');
end
xlabel('Step');

figure('Name','Encoder RAW');
for i = 1:4
    subplot(4,1,i)
    plot(t,enc(:,i),'k','LineWidth',1.5);
    grid on;
    ylabel(['M',num2str(i)]);
    title(['Motor ',num2str(i),' encoder RAW']);
end
xlabel('Step');

fprintf('\n===== Campos dentro de encoderDiag =====\n');
disp(fieldnames(D));

fprintf('\n===== Correlacion q vs q_ref =====\n');
for i = 1:4
    c = corr(q(:,i), q_ref(:,i), 'Rows','complete');
    fprintf('Motor %d: corr = %.3f\n', i, c);
end

fprintf('\n===== MAE q vs q_ref =====\n');
for i = 1:4
    mae = mean(abs(q(:,i)-q_ref(:,i)), 'omitnan');
    fprintf('Motor %d: MAE = %.3f\n', i, mae);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maeGlobal = mean(abs(q - q_ref), 'all', 'omitnan');
fprintf('\nMAE global q vs q_ref = %.4f\n', maeGlobal);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n===== CORRELACION Y MAE ALINEADOS =====\n');

for i = 1:4
    corrAligned = corr(q_aligned(:,i), q_ref_aligned(:,i), 'Rows','complete');
    maeAligned  = mean(abs(q_aligned(:,i) - q_ref_aligned(:,i)), 'omitnan');

    fprintf('Motor %d: corr aligned = %.3f | MAE aligned = %.3f\n', ...
        i, corrAligned, maeAligned);
end


fprintf('\n===== CROSS-CORRELATION TEMPORAL =====\n');

for i = 1:4

    x = q_aligned(:,i);
    y = q_ref_aligned(:,i);

    x = x - mean(x,'omitnan');
    y = y - mean(y,'omitnan');

    [c,lags] = xcorr(x, y, 'coeff');

    [maxCorr, idx] = max(c);
    bestLag = lags(idx);

    fprintf('Motor %d: max xcorr = %.3f | best lag = %d steps\n', ...
        i, maxCorr, bestLag);
end


%% ===== Prueba de ganancia por motor =====
gain = zeros(1,4);

for i = 1:4
    den = sum(q_ref(:,i).^2);
    if den > 1e-9
        gain(i) = sum(q(:,i).*q_ref(:,i)) / den;
    else
        gain(i) = 1;
    end
end

q_ref_gain = q_ref .* gain;
q_ref_gain = max(0, min(1, q_ref_gain));

fprintf('\n===== GANANCIA ESTIMADA POR MOTOR =====\n');
disp(gain);


figure('Name','q vs q_ref con offset + ganancia');

for i = 1:4
    subplot(4,1,i)

    plot(t, q(:,i), 'b', 'LineWidth', 1.5);
    hold on
    plot(t, q_ref_gain(:,i), 'r', 'LineWidth', 1.5);

    grid on
    ylabel(['M',num2str(i)])
    title(['Motor ',num2str(i),' - q vs q\_ref gain'])

    legend('q encoder','q\_ref gain')
end

xlabel('Step')




fprintf('\n===== MAE CON OFFSET + GANANCIA =====\n');

for i = 1:4
    maeGain = mean(abs(q(:,i)-q_ref_gain(:,i)), 'omitnan');
    corrGain = corr(q(:,i), q_ref_gain(:,i), 'Rows','complete');

    fprintf('Motor %d: corr = %.3f | MAE gain = %.3f\n', ...
        i, corrGain, maeGain);
end

maeGlobalGain = mean(abs(q - q_ref_gain), 'all', 'omitnan');
fprintf('MAE global con ganancia = %.4f\n', maeGlobalGain);





outPlotDir = fullfile(path, "C:\trainedAgentsProtesisNew\00_oldy\_\plots");
if ~exist(outPlotDir, "dir")
    mkdir(outPlotDir);
end

figs = findall(0, 'Type', 'figure');

for k = 1:length(figs)
    saveas(figs(k), fullfile(outPlotDir, sprintf('fig_%02d.jpg', k)));
end

fprintf('\nFiguras guardadas en: %s\n', outPlotDir);