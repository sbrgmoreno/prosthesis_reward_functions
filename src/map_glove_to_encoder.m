function qrefCorr = map_glove_to_encoder(flexRefMat, calib)
% Mapea referencia del glove al espacio del encoder
% usando:
% - permutacion de canales
% - inversion
% - ajuste lineal
%
% Entrada:
%   flexRefMat : Nx4 o 1x4
%   calib      : estructura con calib.perMotor
%
% Salida:
%   qrefCorr   : Nx4

    if isempty(flexRefMat)
        qrefCorr = [];
        return;
    end

    if isvector(flexRefMat)
        flexRefMat = reshape(flexRefMat, 1, []);
    end

    N = size(flexRefMat,1);
    nMotors = numel(calib.perMotor);

    qrefCorr = nan(N, nMotors);

    for i = 1:nMotors
        m = calib.perMotor(i);

        x = flexRefMat(:, m.sourceIdx);

        if m.invert
            x = 1 - x;
        end

        yHat = m.a * x + m.b;
        yHat = max(0, min(1, yHat));

        qrefCorr(:,i) = yHat;
    end
end
