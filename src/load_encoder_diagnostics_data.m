function [allQ, allQref, allEncRaw] = load_encoder_diagnostics_data(folderPath)
% ============================================================
% Carga todos los archivos encoder_diag_ep_*.mat
% y construye:
%
% allQ        -> adjustEncLog (encoder normalizado)
% allQref     -> flexConvertedLog (referencia)
% allEncRaw   -> encRawLog (encoder crudo)
%
% Uso:
% [allQ, allQref, allEncRaw] = load_encoder_diagnostics_data(folderPath)
% ============================================================

    if nargin < 1 || isempty(folderPath)
        folderPath = "encoder_diagnostics";
    end

    files = dir(fullfile(folderPath, "encoder_diag_ep_*.mat"));

    if isempty(files)
        error("No se encontraron archivos encoder_diag_ep_*.mat en %s", folderPath);
    end

    % Inicializar acumuladores
    allQ = [];
    allQref = [];
    allEncRaw = [];

    fprintf("\n========================================\n");
    fprintf(" CARGANDO DATOS DE ENCODER\n");
    fprintf(" Carpeta: %s\n", folderPath);
    fprintf(" Archivos encontrados: %d\n", numel(files));
    fprintf("========================================\n");

    for k = 1:numel(files)
        filePath = fullfile(folderPath, files(k).name);

        S = load(filePath);

        if ~isfield(S, "encoderDiag")
            warning("Archivo %s no contiene encoderDiag. Se omite.", files(k).name);
            continue;
        end

        D = S.encoderDiag;

        % Verificar campos necesarios
        if ~isfield(D, "adjustEncLog") || ...
           ~isfield(D, "flexConvertedLog") || ...
           ~isfield(D, "encRawLog")

            warning("Archivo %s no tiene todos los campos requeridos. Se omite.", files(k).name);
            continue;
        end

        q = D.adjustEncLog;
        qref = D.flexConvertedLog;
        encRaw = D.encRawLog;

        % Alinear tamaños
        n = min([size(q,1), size(qref,1), size(encRaw,1)]);

        if n == 0
            warning("Archivo %s tiene datos vacíos. Se omite.", files(k).name);
            continue;
        end

        q = q(1:n,:);
        qref = qref(1:n,:);
        encRaw = encRaw(1:n,:);

        % Eliminar filas con NaN
        validRows = ...
            ~any(isnan(q),2) & ...
            ~any(isnan(qref),2) & ...
            ~any(isnan(encRaw),2);

        q = q(validRows,:);
        qref = qref(validRows,:);
        encRaw = encRaw(validRows,:);

        % Acumular
        allQ = [allQ; q];
        allQref = [allQref; qref];
        allEncRaw = [allEncRaw; encRaw];
    end

    if isempty(allQ) || isempty(allQref)
        error("No se pudieron acumular datos válidos.");
    end

    fprintf("\nDatos cargados correctamente:\n");
    fprintf("  allQ     : %d x %d\n", size(allQ,1), size(allQ,2));
    fprintf("  allQref  : %d x %d\n", size(allQref,1), size(allQref,2));
    fprintf("  allEncRaw: %d x %d\n", size(allEncRaw,1), size(allEncRaw,2));
end
