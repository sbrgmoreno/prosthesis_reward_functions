%% analisis_emgs_gloves.m

clear
clc

load("DENIS.mat")

fprintf("\n=========================================\n");
fprintf("ANALISIS DATASET DENIS\n");
fprintf("=========================================\n");

%% Variables principales

fprintf("\n===== VARIABLES =====\n");
whos

fprintf("\n===== TAMAÑOS =====\n");
fprintf("size(emgs)   = %s\n", mat2str(size(emgs)));
fprintf("size(gloves) = %s\n", mat2str(size(gloves)));

%% Longitudes

numRows = size(emgs,1);
numCols = size(emgs,2);

emgLengths   = [];
gloveLengths = [];

fprintf("\n===== RECORRIENDO DATASET =====\n");

for i = 1:numRows
    for j = 1:numCols

        E = emgs{i,j};
        G = gloves{i,j};

        if isempty(E) || isempty(G)
            continue
        end

        emgLengths(end+1)   = size(E,1);
        gloveLengths(end+1) = length(G);

    end
end

%% Estadísticas EMG

fprintf("\n===== EMG =====\n");

fprintf("Total registros: %d\n", length(emgLengths));

fprintf("Min samples : %d\n", min(emgLengths));
fprintf("Max samples : %d\n", max(emgLengths));
fprintf("Mean samples: %.2f\n", mean(emgLengths));
fprintf("Std samples : %.2f\n", std(emgLengths));

%% Estadísticas Glove

fprintf("\n===== GLOVE =====\n");

fprintf("Total registros: %d\n", length(gloveLengths));

fprintf("Min samples : %d\n", min(gloveLengths));
fprintf("Max samples : %d\n", max(gloveLengths));
fprintf("Mean samples: %.2f\n", mean(gloveLengths));
fprintf("Std samples : %.2f\n", std(gloveLengths));

%% Relación de frecuencias

ratio = emgLengths ./ gloveLengths;

fprintf("\n===== EMG/GLOVE RATIO =====\n");

fprintf("Min ratio  : %.2f\n", min(ratio));
fprintf("Max ratio  : %.2f\n", max(ratio));
fprintf("Mean ratio : %.2f\n", mean(ratio));
fprintf("Std ratio  : %.2f\n", std(ratio));

%% Histograma

figure
histogram(emgLengths)
title('EMG lengths')
xlabel('Samples')
ylabel('Count')
grid on

figure
histogram(gloveLengths)
title('Glove lengths')
xlabel('Samples')
ylabel('Count')
grid on

figure
histogram(ratio)
title('EMG / Glove ratio')
xlabel('Ratio')
ylabel('Count')
grid on

%% Ejemplo de registro

fprintf("\n===== EJEMPLO =====\n");

E = emgs{1,1};
G = gloves{1,1};

fprintf("EMG size   : %s\n", mat2str(size(E)));
fprintf("Glove size : %d\n", length(G));

disp("Campos glove:")
disp(fieldnames(G(1)))

fprintf("\nANALISIS COMPLETADO\n");