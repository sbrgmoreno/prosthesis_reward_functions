function state = calculateState(this, emg, motorData)
% calculateState returns the normalized state:
% [EMG features; current prosthesis state q; tracking error err = q_ref - q; dq]

    % 1) EMG features
    emg = this.featureCalculator(emg);   % expected 40x1

    % 2) q desde encoder crudo normalizado por motor
    encRawMat = motorData;

    % Rangos calibrados provisionalmente
    encMin = [0 0 -5 -10];
    encMax = [250 320 120 340];

    qMat = (encRawMat - encMin) ./ (encMax - encMin);
    qMat = max(0, min(1, qMat));
    q = qMat(end,:)';   % 4x1

    % 3) referencia q_ref desde glove
    if isempty(this.flexData)
        q_ref = zeros(4,1);
    else
        flexRef = this.flexJoined_scaler(reduceFlexDimension(this.flexData)); % Nx4
        flexRef = max(0, min(1, flexRef));
        q_ref = flexRef(end,:)';
    end

    % 4) error
    err = q_ref - q;
    err = max(-1, min(1, err));

    % 5) dq = q(t) - q(t-1)
    if isempty(this.prevQ)
        dq = zeros(4,1);
    else
        dq = q - this.prevQ;
    end

    % actualizar memoria
    this.prevQ = q;

    % 6) estado final
    state = [emg; q; err; dq];

    % 7) safety check
    a = this.getObservationInfo;
    try
        assert(all(state >= a.LowerLimit) && all(state <= a.UpperLimit), ...
            'state outside of range')
    catch
        this.prosthesis.stop();

        fprintf('\n[STATE DEBUG]\n');
        fprintf('min(state-a.LowerLimit)=%.6f\n', min(state - a.LowerLimit));
        fprintf('max(state-a.UpperLimit)=%.6f\n', max(state - a.UpperLimit));

        fprintf('q = %s\n', mat2str(q',4));
        fprintf('q_ref = %s\n', mat2str(q_ref',4));
        fprintf('err = %s\n', mat2str(err',4));
        fprintf('dq = %s\n', mat2str(dq',4));

        error('state outside of range');
    end
end











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