clear
clc
close all

fprintf('\n====================================\n');
fprintf('EMG -> GLOVE ALIGNMENT\n');
fprintf('====================================\n');

load('DENIS.mat')

%% ============================================
% Inicialización
%% ============================================

params = configurables();

featureFcn = params.fGetFeatures;

X = [];
Y = [];

numPairs = size(emgs,1);

fprintf('\nProcesando %d pares...\n',numPairs);

%% ============================================
% Recorrer dataset
%% ============================================

for i = 1:numPairs

    try

        emgData = emgs{i,1};

        gloveStruct = gloves{i,1};

        %% =====================================
        % Reducir glove
        %% =====================================

        gloveReduced = reduceFlexDimension(gloveStruct);

        if isempty(gloveReduced)
            continue
        end

        Nemg = size(emgData,1);
        Nglove = size(gloveReduced,1);

        %% =====================================
        % Tiempo normalizado
        %% =====================================

        tEMG   = linspace(0,1,Nemg);
        tGlove = linspace(0,1,Nglove);

        %% =====================================
        % Para cada muestra glove
        %% =====================================

        for k = 1:Nglove

            tg = tGlove(k);

            idx = find(abs(tEMG - tg) == min(abs(tEMG - tg)),1);

            %% ventana EMG

            win = 200;

            i0 = max(1,idx-win+1);
            i1 = idx;

            emgWindow = emgData(i0:i1,:);

            %% mínimo tamaño

            if size(emgWindow,1) < 50
                continue
            end

            %% features

            feat = featureFcn(emgWindow);

            feat = feat(:)';

            X = [X; feat];

            Y = [Y; gloveReduced(k,:)];

        end

    catch ME

        fprintf('Error en muestra %d\n',i);
        fprintf('%s\n',ME.message);

    end

end

%% ============================================
% Resumen
%% ============================================

fprintf('\n===== DATASET FINAL =====\n');

fprintf('X size = [%d %d]\n',size(X,1),size(X,2));
fprintf('Y size = [%d %d]\n',size(Y,1),size(Y,2));

save EMG_GLOVE_DATASET.mat X Y

fprintf('\nDataset guardado.\n');