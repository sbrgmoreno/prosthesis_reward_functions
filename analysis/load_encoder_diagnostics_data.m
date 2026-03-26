function [allQ, allQref, allEncRaw] = load_encoder_diagnostics_data(folderPath)
% Carga todos los archivos encoder_diag_ep_*.mat y acumula:
% allQ     -> adjustEncLog
% allQref  -> flexConvertedLog
% allEncRaw -> encRawLog

    if nargin < 1 || isempty(folderPath)
        folderPath = "encoder_diagnostics";
    end

    files = dir(fullfile(folderPath, "encoder_diag_ep_*.mat"));

    if isempty(files)
        error("No se encontraron archivos encoder_diag_ep_*.mat en %s", folderPath);
    end

    allQ = [];
    allQref = [];
    allEncRaw = [];

    for k = 1:numel(files)
        S = load(fullfile(folderPath, files(k).name));

        if ~isfield(S, "encoderDiag")
            warning("Archivo %s no contiene encoderDiag. Se omite.", files(k).name);
            continue;
        end

        D = S.encoderDiag;

        if ~isfield(D, "adjustEncLog") || ~isfield(D, "flexConvertedLog") || ~isfield(D, "encRawLog")
            warning("Archivo %s no contiene todos los campos requeridos. Se omite.", files(k).name);
            continue;
        end

        q = D.adjustEncLog;
        qref = D.flexConvertedLog;
        encRaw = D.encRawLog;

        n = min([size(q,1), size(qref,1), size(encRaw,1)]);
        if n == 0
            continue;
        end

        q = q(1:n,:);
        qref = qref(1:n,:);
        encRaw = encRaw(1:n,:);

        % quitar filas con NaN completos si existen
        validRows = ~any(isnan(q),2) & ~any(isnan(qref),2) & ~any(isnan(encRaw),2);

        q = q(validRows,:);
        qref = qref(validRows,:);
        encRaw = encRaw(validRows,:);

        allQ = [allQ; q];
        allQref = [allQref; qref];
        allEncRaw = [allEncRaw; encRaw];
    end

    if isempty(allQ) || isempty(allQref)
        error("No se pudieron cargar datos válidos.");
    end

    fprintf("\nDatos cargados correctamente:\n");
    fprintf("  allQ     : %d x %d\n", size(allQ,1), size(allQ,2));
    fprintf("  allQref  : %d x %d\n", size(allQref,1), size(allQref,2));
    fprintf("  allEncRaw: %d x %d\n", size(allEncRaw,1), size(allEncRaw,2));
end

