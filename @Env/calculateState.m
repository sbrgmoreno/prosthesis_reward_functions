function state = calculateState(this, emg, motorData)
% calculateState returns the normalized state:
% [EMG features; current prosthesis state q; tracking error err = q_ref - q]

% 1) EMG features
emg = this.featureCalculator(emg);   % 40x1

% 2) q consistente con step.m
enc = motorData(end,:);
encMin = [0 0 0 0];
encMax = [5000 5000 5000 5000];
q = max(0, min(1, (enc - encMin) ./ (encMax - encMin)));
q = q(:);   % columna

% 3) Current reference q_ref from glove/dataset
if isempty(this.flexData)
    q_ref = zeros(4,1);
else
    flexRef = this.flexJoined_scaler(reduceFlexDimension(this.flexData));  % Nx4
    q_ref = flexRef(end, :)';
    q_ref = max(0, min(1, q_ref));   % CLIPPING SEGURO
end

% 4) Tracking error
err = q_ref - q;
err = max(-1, min(1, err));   % CLIPPING SEGURO

% 5) Final state
state = [emg; q; err];

% 6) Safety check
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

    error('state outside of range');
end
end




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