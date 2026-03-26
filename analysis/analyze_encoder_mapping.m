function analyze_encoder_mapping(folderPath)

if nargin < 1
    folderPath = "encoder_diagnostics";
end

files = dir(fullfile(folderPath, "encoder_diag_ep_*.mat"));

allQ = [];
allQref = [];

for k = 1:numel(files)
    S = load(fullfile(folderPath, files(k).name));
    D = S.encoderDiag;

    n = min(size(D.adjustEncLog,1), size(D.flexConvertedLog,1));

    q = D.adjustEncLog(1:n,:);
    qref = D.flexConvertedLog(1:n,:);

    allQ = [allQ; q];
    allQref = [allQref; qref];
end

fprintf("\n=== MATRIZ DE CORRELACION (sin invertir) ===\n");

C = zeros(4,4);

for i = 1:4
    for j = 1:4
        C(i,j) = corr(allQ(:,i), allQref(:,j), 'Rows','complete');
    end
end

disp(C);

fprintf("\n=== MATRIZ DE CORRELACION (invertido) ===\n");

Cinv = zeros(4,4);

for i = 1:4
    for j = 1:4
        Cinv(i,j) = corr(1 - allQ(:,i), allQref(:,j), 'Rows','complete');
    end
end

disp(Cinv);

