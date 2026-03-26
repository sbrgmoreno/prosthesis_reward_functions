function calib = fit_qref_calibration(allQ, allQref, maxLag)
% Busca para cada motor:
% - mejor canal de referencia
% - si conviene invertirlo
% - mejor lag
% - ajuste lineal y = a*x + b

    if nargin < 3 || isempty(maxLag)
        maxLag = 20;
    end

    nMotors = size(allQ,2);
    lags = -maxLag:maxLag;

    calib = struct();

    emptyBest = struct( ...
        'targetMotor', [], ...
        'sourceIdx', [], ...
        'invert', [], ...
        'lag', [], ...
        'a', [], ...
        'b', [], ...
        'corrBefore', [], ...
        'corrAfter', [] ...
    );

    calib.perMotor = repmat(emptyBest, 1, nMotors);

    for i = 1:nMotors
        bestScore = -Inf;

        best = emptyBest;
        best.targetMotor = i;
        best.sourceIdx = NaN;
        best.invert = false;
        best.lag = 0;
        best.a = 1;
        best.b = 0;
        best.corrBefore = NaN;
        best.corrAfter = NaN;

        yFull = allQ(:,i);   % encoder target

        for j = 1:nMotors
            xBase = allQref(:,j);

            for invFlag = [false true]
                if invFlag
                    xCand = 1 - xBase;
                else
                    xCand = xBase;
                end

                for lag = lags
                    [xAligned, yAligned] = align_with_lag(xCand, yFull, lag);

                    valid = ~isnan(xAligned) & ~isnan(yAligned);
                    x = xAligned(valid);
                    y = yAligned(valid);

                    if numel(x) < 10
                        continue;
                    end

                    c0 = corr(x, y, 'Rows', 'complete');
                    if isnan(c0)
                        continue;
                    end

                    % Ajuste lineal y = a*x + b
                    p = polyfit(x, y, 1);
                    a = p(1);
                    b = p(2);

                    yHat = a*x + b;
                    yHat = max(0, min(1, yHat));

                    c1 = corr(yHat, y, 'Rows', 'complete');
                    if isnan(c1)
                        c1 = -Inf;
                    end

                    score = c1;

                    if score > bestScore
                        bestScore = score;
                        best.sourceIdx = j;
                        best.invert = invFlag;
                        best.lag = lag;
                        best.a = a;
                        best.b = b;
                        best.corrBefore = c0;
                        best.corrAfter = c1;
                    end
                end
            end
        end

        calib.perMotor(i) = best;
    end

    fprintf('\n========================================\n');
    fprintf(' CALIBRACION q_ref -> encoder\n');
    fprintf('========================================\n');

    for i = 1:nMotors
        m = calib.perMotor(i);
        fprintf('\nMotor destino %d\n', i);
        fprintf('  sourceIdx   = %d\n', m.sourceIdx);
        fprintf('  invert      = %d\n', m.invert);
        fprintf('  lag         = %d\n', m.lag);
        fprintf('  a           = %.6f\n', m.a);
        fprintf('  b           = %.6f\n', m.b);
        fprintf('  corrBefore  = %.6f\n', m.corrBefore);
        fprintf('  corrAfter   = %.6f\n', m.corrAfter);
    end
end

function [xAligned, yAligned] = align_with_lag(x, y, lag)
% lag > 0  => x se retrasa respecto a y
% lag < 0  => x se adelanta respecto a y

    if lag > 0
        xAligned = x(1:end-lag);
        yAligned = y(1+lag:end);
    elseif lag < 0
        xAligned = x(1-lag:end);
        yAligned = y(1:end+lag);
    else
        xAligned = x;
        yAligned = y;
    end
end
