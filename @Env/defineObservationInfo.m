
% %%%%%%%%%%%% 88 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function obsInfo = defineObservationInfo()
% % Observation with short history:
% %   EMG(t) = 40
% %   History of 3 blocks:
% %       [q; q_ref_pred; err_pred; dq] = 16
% %   Total = 40 + 3*16 = 88
% 
% hardware = definitions();
% 
% numEMGFeatures = configurables("numEMGFeatures"); % 40
% numMotors = hardware.numMotors;                   % 4
% histLen = 3;
% 
% kinBlockLength = 4*numMotors;                     % q, q_ref, err, dq = 16
% stateLength = numEMGFeatures + histLen*kinBlockLength;
% 
% EMGFeaturesMin = -inf;
% EMGFeaturesMax = inf;
% 
% qMin = 0;
% qMax = 1;
% 
% qRefMin = 0;
% qRefMax = 1;
% 
% errMin = -1;
% errMax = 1;
% 
% dqMin = -1;
% dqMax = 1;
% 
% kinLowerBlock = [
%     qMin    * ones(numMotors,1);
%     qRefMin * ones(numMotors,1);
%     errMin  * ones(numMotors,1);
%     dqMin   * ones(numMotors,1)
% ];
% 
% kinUpperBlock = [
%     qMax    * ones(numMotors,1);
%     qRefMax * ones(numMotors,1);
%     errMax  * ones(numMotors,1);
%     dqMax   * ones(numMotors,1)
% ];
% 
% obsInfo = rlNumericSpec([stateLength 1]);
% 
% obsInfo.LowerLimit = [
%     EMGFeaturesMin * ones(numEMGFeatures,1);
%     repmat(kinLowerBlock, histLen, 1)
% ];
% 
% obsInfo.UpperLimit = [
%     EMGFeaturesMax * ones(numEMGFeatures,1);
%     repmat(kinUpperBlock, histLen, 1)
% ];
% 
% obsInfo.Name = 'prosthesis_state_88_hist';
% obsInfo.Description = sprintf( ...
%     'State with %d EMG features and %d-step history of [q qref err dq]', ...
%     numEMGFeatures, histLen);
% 
% assert(all(size(obsInfo.LowerLimit) == [stateLength 1]), ...
%     'LowerLimit size mismatch with stateLength');
% 
% assert(all(size(obsInfo.UpperLimit) == [stateLength 1]), ...
%     'UpperLimit size mismatch with stateLength');
% 
% end





% %%%%%% 52 features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function obsInfo = defineObservationInfo()
% % Observation:
% %   - EMG features
% %   - current prosthesis state q
% %   - error err = q_ref - q
% %   - delta state dq = q_t - q_(t-1)
% 
% %% aux vars
% hardware = definitions();
% 
% numEMGFeatures = configurables("numEMGFeatures"); % 40
% numMotors = hardware.numMotors;                   % 4
% 
% % Estado: EMG + q + err + dq
% stateLength = numEMGFeatures + 3*numMotors;        % 40 + 12 = 52
% 
% % Ranges
% EMGFeaturesMin = -inf;
% EMGFeaturesMax = inf;
% 
% qMin = 0;
% qMax = 1;
% 
% errMin = -1;
% errMax = 1;
% 
% dqMin = -1;
% dqMax = 1;
% 
% %% creating observation space
% obsInfo = rlNumericSpec([stateLength 1]);
% 
% %% limits
% obsInfo.LowerLimit = [ ...
%     EMGFeaturesMin * ones(numEMGFeatures, 1); ...
%     qMin           * ones(numMotors, 1); ...
%     errMin         * ones(numMotors, 1); ...
%     dqMin          * ones(numMotors, 1)];
% 
% obsInfo.UpperLimit = [ ...
%     EMGFeaturesMax * ones(numEMGFeatures, 1); ...
%     qMax           * ones(numMotors, 1); ...
%     errMax         * ones(numMotors, 1); ...
%     dqMax          * ones(numMotors, 1)];
% 
% obsInfo.Name = 'prosthesis_state_52';
% obsInfo.Description = sprintf( ...
%     ['State defined with %d EMG features, %d q values, ', ...
%      '%d err values, and %d dq values'], ...
%     numEMGFeatures, numMotors, numMotors, numMotors);
% 
% %% safety check
% assert(all(size(obsInfo.LowerLimit) == [stateLength 1]), ...
%     'LowerLimit size mismatch with stateLength');
% 
% assert(all(size(obsInfo.UpperLimit) == [stateLength 1]), ...
%     'UpperLimit size mismatch with stateLength');
% 
% end

%%%%%% 56 features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function obsInfo = defineObservationInfo()
% Observation:
%   - EMG features
%   - current prosthesis state q
%   - weak reference q_ref
%   - error err = q_ref - q
%   - delta state dq = q_t - q_(t-1)

%% aux vars
hardware = definitions();

numEMGFeatures = configurables("numEMGFeatures"); % 40
numMotors = hardware.numMotors;                   % 4

% Nuevo estado: EMG + q + q_ref + err + dq
stateLength = numEMGFeatures + 4*numMotors;        % 40 + 16 = 56

% Ranges
EMGFeaturesMin = -inf;
EMGFeaturesMax = inf;

qMin = 0;
qMax = 1;

qRefMin = 0;
qRefMax = 1;

errMin = -1;
errMax = 1;

dqMin = -1;
dqMax = 1;

%% creating observation space
obsInfo = rlNumericSpec([stateLength 1]);

%% limits
obsInfo.LowerLimit = [ ...
    EMGFeaturesMin * ones(numEMGFeatures, 1); ...
    qMin           * ones(numMotors, 1); ...
    qRefMin        * ones(numMotors, 1); ...
    errMin         * ones(numMotors, 1); ...
    dqMin          * ones(numMotors, 1)];

obsInfo.UpperLimit = [ ...
    EMGFeaturesMax * ones(numEMGFeatures, 1); ...
    qMax           * ones(numMotors, 1); ...
    qRefMax        * ones(numMotors, 1); ...
    errMax         * ones(numMotors, 1); ...
    dqMax          * ones(numMotors, 1)];

obsInfo.Name = 'prosthesis_state_56';
obsInfo.Description = sprintf( ...
    ['State defined with %d EMG features, %d q values, ', ...
     '%d q_ref values, %d err values, and %d dq values'], ...
    numEMGFeatures, numMotors, numMotors, numMotors, numMotors);

%% safety check
assert(all(size(obsInfo.LowerLimit) == [stateLength 1]), ...
    'LowerLimit size mismatch with stateLength');

assert(all(size(obsInfo.UpperLimit) == [stateLength 1]), ...
    'UpperLimit size mismatch with stateLength');

end


%%%%%% 48 features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function obsInfo = defineObservationInfo()
% % defineObservationInfo() returns the limits and dimension of the
% % observation of the environment.
% % The observation is defined as the concatenation of:
% %   - EMG features
% %   - current prosthesis state q
% %   - delta state dq = q_t - q_(t-1)
% 
% %% aux vars
% params = configurables();
% hardware = definitions();
% 
% numEMGFeatures = configurables("numEMGFeatures");
% numMotors = hardware.numMotors;
% stateLength = configurables("stateLength");
% 
% % Ranges
% EMGFeaturesMin = -inf;
% EMGFeaturesMax = inf;
% 
% % q normalized
% qMin = 0;
% qMax = 1;
% 
% % dq = q_t - q_(t-1), with q in [0,1]
% dqMin = -1;
% dqMax = 1;
% 
% %% creating observation space
% obsInfo = rlNumericSpec([stateLength 1]); % col-wise
% 
% %% limits
% obsInfo.LowerLimit = [ ...
%     EMGFeaturesMin * ones(numEMGFeatures, 1); ...
%     qMin    * ones(numMotors, 1); ...
%     dqMin   * ones(numMotors, 1)];
% 
% obsInfo.UpperLimit = [ ...
%     EMGFeaturesMax * ones(numEMGFeatures, 1); ...
%     qMax    * ones(numMotors, 1); ...
%     dqMax   * ones(numMotors, 1)];
% 
% obsInfo.Name = 'prosthesis_state';
% obsInfo.Description = sprintf( ...
%     'State defined with %d EMG features, %d encoder states, and %d dq values', ...
%     numEMGFeatures, numMotors, numMotors);
% 
% % Optional safety check
% assert(all(size(obsInfo.LowerLimit) == [stateLength 1]), ...
%     'LowerLimit size mismatch with stateLength');
% assert(all(size(obsInfo.UpperLimit) == [stateLength 1]), ...
%     'UpperLimit size mismatch with stateLength');
% 
% end



% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 52 INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function obsInfo = defineObservationInfo()
% % defineObservationInfo() returns the limits and dimension of the
% % observation of the environment.
% % The observation is defined as the concatenation of:
% %   - EMG features
% %   - current prosthesis state q
% %   - tracking error err = q_ref - q
% %   - delta state dq = q_t - q_(t-1)
% 
% %% aux vars
% params = configurables();
% hardware = definitions();
% 
% numEMGFeatures = configurables("numEMGFeatures");
% numMotors = hardware.numMotors;
% stateLength = configurables("stateLength");
% 
% % Ranges
% EMGFeaturesMin = -inf;
% EMGFeaturesMax = inf;
% 
% % q normalized
% qMin = 0;
% qMax = 1;
% 
% % err = q_ref - q, assuming both normalized in [0,1]
% errMin = -1;
% errMax = 1;
% 
% % dq = q_t - q_(t-1), with q in [0,1]
% dqMin = -1;
% dqMax = 1;
% 
% %% creating observation space
% obsInfo = rlNumericSpec([stateLength 1]); % col-wise
% 
% %% limits
% obsInfo.LowerLimit = [ ...
%     EMGFeaturesMin * ones(numEMGFeatures, 1); ...
%     qMin   * ones(numMotors, 1); ...
%     errMin * ones(numMotors, 1); ...
%     dqMin  * ones(numMotors, 1)];
% 
% obsInfo.UpperLimit = [ ...
%     EMGFeaturesMax * ones(numEMGFeatures, 1); ...
%     qMax   * ones(numMotors, 1); ...
%     errMax * ones(numMotors, 1); ...
%     dqMax  * ones(numMotors, 1)];
% 
% obsInfo.Name = 'prosthesis_state';
% obsInfo.Description = sprintf( ...
%     'State defined with %d EMG features, %d encoder states, %d tracking errors, and %d dq values', ...
%     numEMGFeatures, numMotors, numMotors, numMotors);
% end

%-----------------------------------------------------------------------------------------------
% function obsInfo = defineObservationInfo()
% % defineObservationInfo() returns the limits and dimension of the
% % observation of the environment.
% % The observation is defined as the concatenation of:
% %   - EMG features
% %   - current prosthesis state q
% %   - tracking error err = q_ref - q
% 
% %% aux vars
% params = configurables();
% hardware = definitions();
% 
% numEMGFeatures = configurables("numEMGFeatures");
% numMotors = hardware.numMotors;
% stateLength = configurables("stateLength");
% 
% % Ranges
% EMGFeaturesMin = -inf;
% EMGFeaturesMax = inf;
% 
% % q normalized
% qMin = 0;
% qMax = 1;
% 
% % err = q_ref - q, assuming both normalized in [0,1]
% errMin = -1;
% errMax = 1;
% 
% %% creating observation space
% obsInfo = rlNumericSpec([stateLength 1]); % col-wise
% 
% %% limits
% obsInfo.LowerLimit = [ ...
%     EMGFeaturesMin * ones(numEMGFeatures, 1); ...
%     qMin   * ones(numMotors, 1); ...
%     errMin * ones(numMotors, 1)];
% 
% obsInfo.UpperLimit = [ ...
%     EMGFeaturesMax * ones(numEMGFeatures, 1); ...
%     qMax   * ones(numMotors, 1); ...
%     errMax * ones(numMotors, 1)];
% 
% obsInfo.Name = 'prosthesis_state';
% obsInfo.Description = sprintf( ...
%     'State defined with %d EMG features, %d encoder states, and %d tracking errors', ...
%     numEMGFeatures, numMotors, numMotors);
% end


%-----------------------------------------------------------------------------------------------

%% ANTES DESCOMENTADO
% function obsInfo = defineObservationInfo()
% %defineObservationInfo() is a static method that retuns the limits and
% %dimension of the observation of the environment.
% %The observation is defined as the concatenation of EMG features with
% %cinematic info. The EMG features is a F-by-1 vector from EMG features.
% %The cinematic info is a 4-by-1 vector with the encoder position of every
% %motor.
% %
% % Examples
% %   obsInfo = Env.defineObservation()
% %
% 
% %{
% Laboratorio de Inteligencia y Visión Artificial
% ESCUELA POLITÉCNICA NACIONAL
% Quito - Ecuador
% 
% autor: ztjona!
% jonathan.a.zea@ieee.org
% 
% "I find that I don't understand things unless I try to program them."
% -Donald E. Knuth
% 
% 12 October 2021
% 
% Mod 2024/jan/3
% %}
% 
% %% aux vars
% %unpacking
% params = configurables();
% hardware = definitions();
% disp(params)
% numEMGFeatures = configurables("numEMGFeatures");
% numMotors = hardware.numMotors;
% 
% stateLength = configurables("stateLength");
% disp("statelength")
% class(stateLength)
% disp("statelength value")
% disp(stateLength)
% 
% farMinEncoderValue = 0;
% farMaxEncoderValue = 350;
% 
% EMGFeaturesMin = -inf;
% EMGFeaturesMax = inf;
% 
% %% creating observation space
% obsInfo = rlNumericSpec([44 1]); % col-wise
% 
% 
% %% limits
% obsInfo.LowerLimit = [EMGFeaturesMin*ones(numEMGFeatures, 1);
%     repmat(farMinEncoderValue, numMotors, 1)];
% 
% obsInfo.UpperLimit = [EMGFeaturesMax*ones(numEMGFeatures, 1);
%     repmat(farMaxEncoderValue, numMotors, 1)];
% 
% obsInfo.Name = 'prosthesis_state';
% obsInfo.Description = sprintf(...
%     'State defined with %d EMG features and %d encoder positions',...
%     numEMGFeatures, numMotors);
