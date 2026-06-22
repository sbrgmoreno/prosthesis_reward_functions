
function state = calculateState(this, emg, motorData)
% calculateState returns the normalized state:
% [EMG features; q; q_ref_pred; err_pred = q_ref_pred - q; dq]
%
% Dimensión:
% 40 EMG + 4 q + 4 q_ref_pred + 4 err_pred + 4 dq = 56

    %% =========================================================
    % 1) EMG features
    % =========================================================
    emg = this.featureCalculator(emg);   % expected 40x1
    emg = emg(:);

    %% =========================================================
    % 2) q actual
    % =========================================================
    if ~isempty(this.adjustEnc)
        q = this.adjustEnc(end,:)';   % 4x1
    else
        encRawMat = motorData;

        encMin = [-10, 0, -1, 0];
        encMax = [175, 235, 485, 185];

        qMat = (encRawMat - encMin) ./ (encMax - encMin);
        qMat = max(0, min(1, qMat));
        q = qMat(end,:)';
    end

    q = max(0, min(1, q));

    %% =========================================================
    % 3) q_ref_pred desde predictor supervisado EMG -> q_ref
    % =========================================================
    % q_ref_pred = predictQrefFromEMG(emg);   % 4x1
    % q_ref_pred = q_ref_pred(:);
    % q_ref_pred = max(0, min(1, q_ref_pred)); %normalizo

    %%%%%%%%%%%%%% suavizacion de las senales de la intencion (prediccion
    %%%%%%%%%%%%%% de EMG 8 canales [] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    q_ref_pred_raw = predictQrefFromEMG(emg);
    q_ref_pred_raw = q_ref_pred_raw(:);
    q_ref_pred_raw = max(0, min(1, q_ref_pred_raw));
    
    alphaQref = 0.15;
    
    if isempty(this.prevQrefPred)
        q_ref_pred = q_ref_pred_raw;
    else
        q_ref_pred = alphaQref * q_ref_pred_raw + ...
                     (1 - alphaQref) * this.prevQrefPred;
    end
    
    q_ref_pred = max(0, min(1, q_ref_pred));
    
    this.prevQrefPred = q_ref_pred;

    %% =========================================================
    % 4) error predicho de intención
    % =========================================================
    err_pred = q_ref_pred - q;
    err_pred = max(-1, min(1, err_pred));

    %% =========================================================
    % 5) dq = q(t) - q(t-1)
    % =========================================================
    if isempty(this.prevQ)
        dq = zeros(4,1);
    else
        dq = q - this.prevQ;
    end

    dq = max(-1, min(1, dq));
    this.prevQ = q;

    %% =========================================================
    % 6) estado final
    % =========================================================
    state = [emg; q; q_ref_pred; err_pred; dq];

    % %% =========================================================
    % % 6) Estado con historial corto
    % % =========================================================
    % % Bloque cinemático/intención actual:
    % % [q; q_ref_pred; err_pred; dq] = 16x1
    % kinNow = [q; q_ref_pred; err_pred; dq];
    % 
    % histLen = 3;
    % 
    % if isempty(this.stateHist)
    %     % Inicializar historial repitiendo el primer estado
    %     this.stateHist = repmat(kinNow, 1, histLen);
    % else
    %     % Desplazar historial:
    %     % columna 1 = actual
    %     % columna 2 = t-1
    %     % columna 3 = t-2
    %     this.stateHist = [kinNow, this.stateHist(:,1:histLen-1)];
    % end
    % 
    % histVec = this.stateHist(:);
    % 
    % state = [emg; histVec];

    %% =========================================================
    % 7) safety check
    % =========================================================
    a = this.getObservationInfo;

    try
        assert(length(state) == length(a.LowerLimit), ...
            'state dimension does not match ObservationInfo')

        assert(all(state >= a.LowerLimit) && all(state <= a.UpperLimit), ...
            'state outside of range')

    catch
        this.prosthesis.stop();

        fprintf('\n[STATE DEBUG]\n');
        fprintf('length(state)=%d\n', length(state));
        fprintf('length(LowerLimit)=%d\n', length(a.LowerLimit));
        fprintf('length(UpperLimit)=%d\n', length(a.UpperLimit));

        if length(state) == length(a.LowerLimit)
            fprintf('min(state-a.LowerLimit)=%.6f\n', min(state - a.LowerLimit));
            fprintf('max(state-a.UpperLimit)=%.6f\n', max(state - a.UpperLimit));
        else
            fprintf('No se puede comparar state con Lower/UpperLimit por mismatch de dimensiones.\n');
        end

        fprintf('q          = %s\n', mat2str(q',4));
        fprintf('q_ref_pred = %s\n', mat2str(q_ref_pred',4));
        fprintf('err_pred   = %s\n', mat2str(err_pred',4));
        fprintf('dq         = %s\n', mat2str(dq',4));

        error('state outside of range');
    end
end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 56 estados %%%%%%%%%%%%%%%%%%%%%%%
% function state = calculateState(this, emg, motorData)
% % calculateState returns the normalized state:
% % [EMG features; q; q_ref; err = q_ref - q; dq]
% %
% % Dimensión:
% % 40 EMG + 4 q + 4 err + 4 dq = 52
% 
%     %% =========================================================
%     % 1) EMG features
%     % =========================================================
%     emg = this.featureCalculator(emg);   % expected 40x1
%     emg = emg(:);
% 
%     %% =========================================================
%     % 2) q actual
%     % =========================================================
%     if ~isempty(this.adjustEnc)
%         q = this.adjustEnc(end,:)';   % 4x1
%     else
%         encRawMat = motorData;
% 
%         encMin = [-10, 0, -1, 0];
%         encMax = [175, 235, 485, 185];
% 
%         qMat = (encRawMat - encMin) ./ (encMax - encMin);
%         qMat = max(0, min(1, qMat));
%         q = qMat(end,:)';
%     end
% 
%     %% =========================================================
%     % 3) q_ref calibrado / señal débil de intención
%     % =========================================================
%     if ~isempty(this.flexConverted)
%         q_ref = this.flexConverted(end,:)';
%     elseif isempty(this.flexData)
%         q_ref = zeros(4,1);
%     else
%         flexRef = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
%         flexRef = max(0, min(1, flexRef));
%         q_ref = flexRef(end,:)';
%     end
% 
%     q_ref = max(0, min(1, q_ref));
% 
%     %% =========================================================
%     % 4) error de intención
%     % =========================================================
%     err = q_ref - q;
%     err = max(-1, min(1, err));
% 
%     %% =========================================================
%     % 5) dq = q(t) - q(t-1)
%     % =========================================================
%     if isempty(this.prevQ)
%         dq = zeros(4,1);
%     else
%         dq = q - this.prevQ;
%     end
% 
%     dq = max(-1, min(1, dq));
% 
%     this.prevQ = q;
% 
%     %% =========================================================
%     % 6) estado final
%     % =========================================================
%     %
%     % calculo de q_ref_pred y err_pred
%     %
% 
%     % q_ref_pred = predictQrefFromEMG(emg);
%     % err_pred = q_ref_pred - q;
%     % err_pred = max(-1, min(1, err_pred));
% 
%     %state = [emg; q; q_ref_pred; err_pred; dq];
%     state = [emg; q; q_ref; err; dq];
%     %state = [emg; q; err; dq];
% 
%     %% =========================================================
%     % 7) safety check
%     % =========================================================
%     a = this.getObservationInfo;
% 
%     try
%         assert(length(state) == length(a.LowerLimit), ...
%             'state dimension does not match ObservationInfo')
% 
%         assert(all(state >= a.LowerLimit) && all(state <= a.UpperLimit), ...
%             'state outside of range')
%     catch
%         this.prosthesis.stop();
% 
%         fprintf('\n[STATE DEBUG]\n');
%         fprintf('length(state)=%d\n', length(state));
%         fprintf('length(LowerLimit)=%d\n', length(a.LowerLimit));
%         fprintf('length(UpperLimit)=%d\n', length(a.UpperLimit));
% 
%         fprintf('min(state-a.LowerLimit)=%.6f\n', min(state - a.LowerLimit));
%         fprintf('max(state-a.UpperLimit)=%.6f\n', max(state - a.UpperLimit));
% 
%         fprintf('q     = %s\n', mat2str(q',4));
%         fprintf('q_ref = %s\n', mat2str(q_ref',4));
%         fprintf('err   = %s\n', mat2str(err',4));
%         fprintf('dq    = %s\n', mat2str(dq',4));
% 
%         error('state outside of range');
%     end
% end





% %%%%%%%%%%%%%%%%%%%%%%%%%%%% 48 INPUT [emg; q; dq] %%%%%%%%%%%
% function state = calculateState(this, emg, motorData)
% % calculateState returns the normalized state:
% % [EMG features; q; q_ref; err = q_ref - q; dq]
% 
%     % =========================================================
%     % 1) EMG features
%     % =========================================================
%     emg = this.featureCalculator(emg);   % expected 40x1
% 
%     % asegurar vector columna
%     emg = emg(:);
% 
%     % =========================================================
%     % 2) q actual (usar adjustEnc si ya fue calculado en step)
%     % =========================================================
%     if ~isempty(this.adjustEnc)
%         q = this.adjustEnc(end,:)';   % 4x1
%     else
%         % fallback por seguridad
%         encRawMat = motorData;
% 
%         %normalizacion con pa/p99
%         encMin = [-10, 0, -1, 0];
%         encMax = [175, 235, 485, 185];
% 
%         qMat = (encRawMat - encMin) ./ (encMax - encMin);
%         qMat = max(0, min(1, qMat));
%         q = qMat(end,:)';
%     end
% 
%     % =========================================================
%     % 3) referencia q_ref desde glove
%     % =========================================================
%     if isempty(this.flexData)
%         q_ref = zeros(4,1);
%     else
%         flexRef = this.flexJoined_scaler(reduceFlexDimension(this.flexData)); % Nx4
%         flexRef = max(0, min(1, flexRef));
%         q_ref = flexRef(end,:)';
%     end
% 
%     % =========================================================
%     % 4) error
%     % =========================================================
%     err = q_ref - q;
%     err = max(-1, min(1, err));
% 
%     % =========================================================
%     % 5) dq = q(t) - q(t-1)
%     % =========================================================
%     if isempty(this.prevQ)
%         dq = zeros(4,1);
%     else
%         dq = q - this.prevQ;
%     end
% 
%     % opcional: limitar dq a rango razonable
%     dq = max(-1, min(1, dq));
% 
%     % actualizar memoria
%     this.prevQ = q;
% 
%     % =========================================================
%     % 6) estado final
%     % nuevo estado: [emg; q; dq]
%     % =========================================================
%     state = [emg; q; dq];
% 
%     % =========================================================
%     % 7) safety check
%     % =========================================================
%     a = this.getObservationInfo;
%     try
%         assert(all(state >= a.LowerLimit) && all(state <= a.UpperLimit), ...
%             'state outside of range')
%     catch
%         this.prosthesis.stop();
% 
%         fprintf('\n[STATE DEBUG]\n');
%         fprintf('min(state-a.LowerLimit)=%.6f\n', min(state - a.LowerLimit));
%         fprintf('max(state-a.UpperLimit)=%.6f\n', max(state - a.UpperLimit));
% 
%         fprintf('q     = %s\n', mat2str(q',4));
%         fprintf('q_ref = %s\n', mat2str(q_ref',4));
%         fprintf('err   = %s\n', mat2str(err',4));
%         fprintf('dq    = %s\n', mat2str(dq',4));
% 
%         error('state outside of range');
%     end
% end




%%%%%%%%%%%%%%%%%%%%%%%%%%%% 52 INPUT [emg; q; err; dq] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function state = calculateState(this, emg, motorData)
% % calculateState returns the normalized state:
% % [EMG features; current prosthesis state q; tracking error err = q_ref - q; dq]
% 
%     % 1) EMG features
%     emg = this.featureCalculator(emg);   % expected 40x1
% 
%     % 2) q desde encoder crudo normalizado por motor
%     encRawMat = motorData;
% 
%     % Rangos calibrados provisionalmente
%     encMin = [0 0 -5 -10];
%     encMax = [250 320 120 340];
% 
%     qMat = (encRawMat - encMin) ./ (encMax - encMin);
%     qMat = max(0, min(1, qMat));
%     q = qMat(end,:)';   % 4x1
% 
%     % 3) referencia q_ref desde glove
%     if isempty(this.flexData)
%         q_ref = zeros(4,1);
%     else
%         flexRef = this.flexJoined_scaler(reduceFlexDimension(this.flexData)); % Nx4
%         flexRef = max(0, min(1, flexRef));
%         q_ref = flexRef(end,:)';
%     end
% 
%     % 4) error
%     err = q_ref - q;
%     err = max(-1, min(1, err));
% 
%     % 5) dq = q(t) - q(t-1)
%     if isempty(this.prevQ)
%         dq = zeros(4,1);
%     else
%         dq = q - this.prevQ;
%     end
% 
%     % actualizar memoria
%     this.prevQ = q;
% 
%     % 6) estado final
%     state = [emg; q; err; dq];
% 
%     % 7) safety check
%     a = this.getObservationInfo;
%     try
%         assert(all(state >= a.LowerLimit) && all(state <= a.UpperLimit), ...
%             'state outside of range')
%     catch
%         this.prosthesis.stop();
% 
%         fprintf('\n[STATE DEBUG]\n');
%         fprintf('min(state-a.LowerLimit)=%.6f\n', min(state - a.LowerLimit));
%         fprintf('max(state-a.UpperLimit)=%.6f\n', max(state - a.UpperLimit));
% 
%         fprintf('q = %s\n', mat2str(q',4));
%         fprintf('q_ref = %s\n', mat2str(q_ref',4));
%         fprintf('err = %s\n', mat2str(err',4));
%         fprintf('dq = %s\n', mat2str(dq',4));
% 
%         error('state outside of range');
%     end
% end











%--------------------------------------------------------------------------------------------

% function state = calculateState(this, emg, motorData)
% % calculateState returns the normalized state:
% % [EMG features; current prosthesis state q; tracking error err = q_ref - q]
% 
%     % 1) EMG features
%     emg = this.featureCalculator(emg);   % 40x1
% 
%     % 2) q desde encoder crudo normalizado por motor
%     encRawMat = motorData;   % Nx4
% 
%     % Ajuste provisional con rangos realistas observados
%     encMin = [0 0 -5 -10];
%     encMax = [250 320 120 340];
% 
%     qMat = (encRawMat - encMin) ./ (encMax - encMin);
%     qMat = max(0, min(1, qMat));
%     q = qMat(end,:)';   % 4x1
% 
%     % 3) referencia q_ref desde glove
%     if isempty(this.flexData)
%         q_ref = zeros(4,1);
%     else
%         flexRef = this.flexJoined_scaler(reduceFlexDimension(this.flexData)); % Nx4
%         flexRef = max(0, min(1, flexRef));
%         q_ref = flexRef(end,:)';
%     end
% 
%     % 4) error
%     err = q_ref - q;
%     err = max(-1, min(1, err));
% 
%     % 5) estado final
%     state = [emg; q; err];
% 
%     % 6) safety check
%     a = this.getObservationInfo;
%     try
%         assert(all(state >= a.LowerLimit) && all(state <= a.UpperLimit), ...
%             'state outside of range')
%     catch
%         this.prosthesis.stop();
% 
%         fprintf('\n[STATE DEBUG]\n');
%         fprintf('min(state-a.LowerLimit)=%.6f\n', min(state - a.LowerLimit));
%         fprintf('max(state-a.UpperLimit)=%.6f\n', max(state - a.UpperLimit));
% 
%         fprintf('q = %s\n', mat2str(q',4));
%         fprintf('q_ref = %s\n', mat2str(q_ref',4));
%         fprintf('err = %s\n', mat2str(err',4));
% 
%         error('state outside of range');
%     end
% end


%--------------------------------------------------------------------------------------------

%{ ANTES DESCOMENTADO
%function state = calculateState(this, emg, motorData)
%obj.calculateState returns the current state of the prosthesis. It
%requires the EMG and cinematic data. The state uses the lattest cinematic
%info. Output is normilized
%
%# Inputs
%
%# Outputs
%* state        -F-by-1 feature state vector. It has EMG feature data and
%               the last motor data
%

% # ---- emg feature extraction. Applies the bag of functions to the emg
% raw signal.
% emg = this.featureCalculator(emg); % E-by-8 -> F-by-1
% 
% enc = this.encoderNormCalculator(motorData(end, :)');
% 
% state = [emg; enc];
% 
% a = this.getObservationInfo;
% try
%     assert(all(state > a.LowerLimit) && all(state < a.UpperLimit), ...
%         'state outside of range')
% catch
%     this.prosthesis.stop();
% end
% end
%}