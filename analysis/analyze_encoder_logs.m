function results = analyze_encoder_logs(folderPath)
% analyze_encoder_logs
% Lee archivos encoder_diag_ep_*.mat y calcula:
% - min/max raw por motor
% - p1/p99 raw por motor
% - % clip a 0 y a 1 en adjustEnc
% - correlacion adjustEnc vs flexConverted por motor
%
% Uso:
%   results = analyze_encoder_logs("C:\ruta\encoder_diagnostics");
%   results = analyze_encoder_logs;  % usa carpeta por defecto

    if nargin < 1 || isempty(folderPath)
        folderPath = "encoder_diagnostics";
    end

    files = dir(fullfile(folderPath, "encoder_diag_ep_*.mat"));

    if isempty(files)
        error("No se encontraron archivos encoder_diag_ep_*.mat en %s", folderPath);
    end

    % ------------------------------------------------------------
    % Acumuladores globales
    % ------------------------------------------------------------
    allEncRaw = [];
    allAdjustEnc = [];
    allFlexConverted = [];

    fprintf("\n========================================\n");
    fprintf(" ANALISIS DE LOGS DE ENCODER\n");
    fprintf(" Carpeta: %s\n", folderPath);
    fprintf(" Archivos encontrados: %d\n", numel(files));
    fprintf("========================================\n");

    % ------------------------------------------------------------
    % Cargar todos los episodios
    % ------------------------------------------------------------
    for k = 1:numel(files)
        S = load(fullfile(folderPath, files(k).name));

        if ~isfield(S, "encoderDiag")
            warning("Archivo %s no contiene encoderDiag. Se omite.", files(k).name);
            continue;
        end

        D = S.encoderDiag;

        % Verificaciones básicas
        if ~isfield(D, "encRawLog") || ~isfield(D, "adjustEncLog") || ~isfield(D, "flexConvertedLog")
            warning("Archivo %s no tiene todos los campos requeridos. Se omite.", files(k).name);
            continue;
        end

        encRaw = D.encRawLog;
        adjEnc = D.adjustEncLog;
        flexRef = D.flexConvertedLog;

        % Alinear por número de filas mínimo
        n = min([size(encRaw,1), size(adjEnc,1), size(flexRef,1)]);
        if n == 0
            warning("Archivo %s tiene logs vacíos. Se omite.", files(k).name);
            continue;
        end

        encRaw = encRaw(1:n, :);
        adjEnc = adjEnc(1:n, :);
        flexRef = flexRef(1:n, :);

        % Acumular
        allEncRaw = [allEncRaw; encRaw];
        allAdjustEnc = [allAdjustEnc; adjEnc];
        allFlexConverted = [allFlexConverted; flexRef];
    end

    if isempty(allEncRaw) || isempty(allAdjustEnc) || isempty(allFlexConverted)
        error("No se pudieron acumular datos válidos.");
    end

    % ------------------------------------------------------------
    % Calcular métricas por motor
    % ------------------------------------------------------------
    nMotors = size(allEncRaw, 2);

    results = struct();
    results.folderPath = folderPath;
    results.numSamples = size(allEncRaw, 1);
    results.numMotors = nMotors;
    
    emptyMotorRes = struct( ...
        'motor', [], ...
        'raw_min', [], ...
        'raw_max', [], ...
        'raw_p1', [], ...
        'raw_p99', [], ...
        'raw_mean', [], ...
        'raw_std', [], ...
        'adjust_min', [], ...
        'adjust_max', [], ...
        'adjust_mean', [], ...
        'adjust_std', [], ...
        'pct_clip_0', [], ...
        'pct_clip_1', [], ...
        'corr_adjust_vs_ref', [] ...
    );
    
    results.perMotor = repmat(emptyMotorRes, 1, nMotors);

    fprintf("\nMuestras acumuladas totales: %d\n", results.numSamples);

    for i = 1:nMotors
        raw_i = allEncRaw(:, i);
        adj_i = allAdjustEnc(:, i);
        ref_i = allFlexConverted(:, i);

        % Eliminar NaN para cálculos robustos
        raw_valid = raw_i(~isnan(raw_i));
        adj_valid = adj_i(~isnan(adj_i));
        ref_valid = ref_i(~isnan(ref_i));

        % Para correlación, alinear por filas válidas conjuntas
        validCorr = ~isnan(adj_i) & ~isnan(ref_i);

        motorRes = emptyMotorRes;
        motorRes.motor = i;
        
        motorRes.raw_min = min(raw_valid);
        motorRes.raw_max = max(raw_valid);
        motorRes.raw_p1 = prctile(raw_valid, 1);
        motorRes.raw_p99 = prctile(raw_valid, 99);
        motorRes.raw_mean = mean(raw_valid);
        motorRes.raw_std = std(raw_valid);
        
        motorRes.adjust_min = min(adj_valid);
        motorRes.adjust_max = max(adj_valid);
        motorRes.adjust_mean = mean(adj_valid);
        motorRes.adjust_std = std(adj_valid);
        
        tol = 1e-9;
        motorRes.pct_clip_0 = 100 * mean(adj_valid <= (0 + tol));
        motorRes.pct_clip_1 = 100 * mean(adj_valid >= (1 - tol));
        
        if sum(validCorr) >= 2
            motorRes.corr_adjust_vs_ref = corr(adj_i(validCorr), ref_i(validCorr), 'Rows', 'complete');
        else
            motorRes.corr_adjust_vs_ref = NaN;
        end
        
        results.perMotor(i) = motorRes;

        % Mostrar resultados
        fprintf("\n----------------------------------------\n");
        fprintf("Motor %d\n", i);
        fprintf("----------------------------------------\n");
        fprintf("Raw min        : %.6f\n", motorRes.raw_min);
        fprintf("Raw max        : %.6f\n", motorRes.raw_max);
        fprintf("Raw p1         : %.6f\n", motorRes.raw_p1);
        fprintf("Raw p99        : %.6f\n", motorRes.raw_p99);
        fprintf("Raw mean       : %.6f\n", motorRes.raw_mean);
        fprintf("Raw std        : %.6f\n", motorRes.raw_std);

        fprintf("Adjust min     : %.6f\n", motorRes.adjust_min);
        fprintf("Adjust max     : %.6f\n", motorRes.adjust_max);
        fprintf("Adjust mean    : %.6f\n", motorRes.adjust_mean);
        fprintf("Adjust std     : %.6f\n", motorRes.adjust_std);

        fprintf("%% clip a 0     : %.2f %%\n", motorRes.pct_clip_0);
        fprintf("%% clip a 1     : %.2f %%\n", motorRes.pct_clip_1);
        fprintf("Corr(adj, ref) : %.6f\n", motorRes.corr_adjust_vs_ref);
    end

    % ------------------------------------------------------------
    % Guardar resultados
    % ------------------------------------------------------------
    outFile = fullfile(folderPath, "encoder_analysis_results.mat");
    save(outFile, "results");

    fprintf("\n========================================\n");
    fprintf("Resultados guardados en:\n%s\n", outFile);
    fprintf("========================================\n");
end