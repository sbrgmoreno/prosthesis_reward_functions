function [observation, reward, isDone, loggedSignals] = step(this, action)

    % Increment step counter
    this.c = this.c + 1;

    % ---- RESET por episodio: inicializar logs por step
    if this.c == 1
        this.meanDistLog     = NaN(1, this.maxNumberStepsInEpisodes);
        this.mseLog          = NaN(1, this.maxNumberStepsInEpisodes);
        this.successLog      = NaN(1, this.maxNumberStepsInEpisodes);
        this.nearSuccessLog  = NaN(1, this.maxNumberStepsInEpisodes);
        
        % reset de memoria para dq en calculateState
        this.prevQ = [];
    end

    % Unify actions if needed (single action -> 4 motors)
    if this.unifyActions
        action = action * [1 1 1 1];
    end

    % Ensure action is numeric row vector
    if iscell(action)
        action = cell2mat(action);
    end
    if ~isnumeric(action)
        error('action must be numeric');
    end

    action = double(action);

    % Expected size for 4 motors
    expectedActionSize = 4;
    if size(action,2) ~= expectedActionSize
        error('The action size does not match the expected number of motors');
    end

    % ===== DEBUG: fuerza acción para test de controlabilidad =====
    if isprop(this,'forceActionDebug') && this.forceActionDebug
        action = this.forcedActionValue;
    end

    % Log raw action
    this.actionLog(this.c,:) = action;
    if this.verbose
        fprintf('  actions=[%2d %2d %2d %2d]\n', action);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (1) SATURACIÓN / CLIPPING DE ACCIÓN
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    actionSat = action;

    if this.rf_modify_actions
        ACTION_MIN = -1;
        ACTION_MAX =  1;
        actionSat = max(min(actionSat, ACTION_MAX), ACTION_MIN);
    end

    this.actionSatLog(this.c,:) = actionSat;

    % Acción aplicada al controlador (con speeds)
    actionApplied = actionSat .* this.speeds;

    fprintf("[DBG] actionSat=%s | speeds=%s | actionApplied=%s\n", ...
        mat2str(actionSat), mat2str(this.speeds), mat2str(actionApplied));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (2) APLICAR ACCIÓN
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow
    completed = this.prosthesis.sendAllSpeed( ...
        actionApplied(1), actionApplied(2), actionApplied(3), actionApplied(4));
    assert(completed, 'ERROR during sending speed to controller')

    while this.periodTic.toc() < this.period
        drawnow
    end

    if this.wait_in_step
        while toc(this.period_realTic) < this.period
            drawnow
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (3) LEER SENSORES / HARDWARE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if this.usePrerecorded
        t_elapsed = this.periodTic.elapsed_time;
        assert(t_elapsed > 0.9*this.period && t_elapsed < 1.1*this.period, ...
            "time elapsed %.2f is incorrect, must be %.2f", ...
            t_elapsed, this.period)

        emg      = this.myo.readEmg(t_elapsed);
        flexData = this.glove.read(t_elapsed);
    else
        emg      = this.myo.readEmg();
        flexData = this.glove.read();
    end

    motorData = this.prosthesis.read();
    this.encoderLog{this.c} = motorData;

    % Encoder crudo para diagnóstico
    encRaw = motorData(end,:);

    % Fallbacks si llegan vacíos
    if isempty(emg)
        emg = this.emg;
        warning("--------------------emg is empty")
    else
        this.emg = emg;
    end

    if isempty(motorData)
        motorData = this.motorData;
        warning("--------------------motorData is empty")
    else
        this.motorData = motorData;
    end

    if isempty(flexData)
        flexData = this.flexData;
        warning("--------------------flexdata is empty")
    else
        this.flexData = flexData;
    end

    % Reset timers
    this.periodTic.tic();
    if this.wait_in_step
        this.period_realTic = tic;
    end

    this.log(sprintf( ...
        '%d. T=%.3f[s]. EmgSize %d. encodersSize %d. glovesize %d', ...
        this.c, this.episodeTic.toc(this.c), ...
        size(emg, 1), size(motorData, 1), size(flexData, 1)));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (4) ACTUALIZAR ESTADO Y VARIABLES AUXILIARES (q y q_ref consistentes)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %--------------------- ANTESDESCOMENTADO---------------------------------------
    % this.flexConverted = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
    % this.adjustEnc     = this.flexJoined_scaler(encoder2Flex(this.motorData));
    % 
    % % clipping seguro
    % this.flexConverted = max(0, min(1, this.flexConverted));
    % this.adjustEnc     = max(0, min(1, this.adjustEnc));
    %---------------------------------------------------------------------
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    this.flexConverted = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
    this.flexConverted = max(0, min(1, this.flexConverted));

    % ===== q desde encoder crudo normalizado por motor =====
    encRawMat = this.motorData;          % Nx4
    encRawLast = encRawMat(end,:);

    % Ajuste provisional con rangos realistas observados
    encMin = [0 0 -5 -10];
    encMax = [250 320 120 340];

    adjEnc = (encRawMat - encMin) ./ (encMax - encMin);
    adjEnc = max(0, min(1, adjEnc));

    this.adjustEnc = adjEnc;

    % ===== DEBUG de transformación de encoder =====
    if this.verbose && (this.c == 1 || mod(this.c,5) == 0)
        fprintf('\n[ENC DEBUG] step=%d\n', this.c);
        fprintf('encRaw(end,:)    = %s\n', mat2str(encRawLast,4));
        fprintf('adjustEnc(end,:) = %s\n', mat2str(this.adjustEnc(end,:),4));
        fprintf('flexRef(end,:)   = %s\n\n', mat2str(this.flexConverted(end,:),4));
    end
    %%%%%%%%%%%%%%%%% 4 LINEAS COMENTADAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    this.State = this.calculateState(emg, motorData);
    observation = this.State;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (5) REWARD
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    reward = 0;
    rewardVector = zeros(1, expectedActionSize);

    try
        [reward, rewardVector, ~] = this.reward_function(this, actionSat);
    catch
        try
            [reward, rewardVector] = this.reward_function(this, actionSat);
        catch
            [reward, rewardVector, ~] = this.reward_function(this, actionSat, []);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (6) MÉTRICAS POR STEP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    q     = this.adjustEnc(end,:);
    q_ref = this.flexConverted(end,:);
    e     = q - q_ref;

    % =========================
    % Sensitivity logging
    % =========================
    t = this.c;

    if t == 1
        this.qLog          = NaN(this.maxNumberStepsInEpisodes, 4);
        this.qRefLog       = NaN(this.maxNumberStepsInEpisodes, 4);
        this.dqLog         = NaN(this.maxNumberStepsInEpisodes, 4);
        this.aRawLog       = NaN(this.maxNumberStepsInEpisodes, 4);
        this.aAppliedLog   = NaN(this.maxNumberStepsInEpisodes, 4);
        this.dirAgreeLog   = false(this.maxNumberStepsInEpisodes, 4);
        this.effectNormLog = NaN(this.maxNumberStepsInEpisodes, 1);
        this.errNormLog    = NaN(this.maxNumberStepsInEpisodes, 1);
        this.dErrLog       = NaN(this.maxNumberStepsInEpisodes, 1);

        this.encRawLog        = NaN(this.maxNumberStepsInEpisodes, 4);
        this.encEffectNormLog = NaN(this.maxNumberStepsInEpisodes, 1);
    end

    this.qLog(t,:)    = q;
    this.qRefLog(t,:) = q_ref;

    this.aRawLog(t,:)     = actionSat;
    this.aAppliedLog(t,:) = actionApplied;

    if t == 1
        dq = zeros(1,4);
    else
        dq = q - this.qLog(t-1,:);
    end
    this.dqLog(t,:) = dq;
    this.effectNormLog(t) = norm(dq);

    this.encRawLog(t,:) = encRaw;
    if t == 1
        dEncRaw = zeros(1,4);
    else
        dEncRaw = encRaw - this.encRawLog(t-1,:);
    end
    this.encEffectNormLog(t) = norm(dEncRaw);

    this.errNormLog(t) = norm(e);

    if t == 1
        this.dErrLog(t) = 0;
    else
        this.dErrLog(t) = this.errNormLog(t-1) - this.errNormLog(t);
    end

    obsDir = sign(dq);
    desiredDQdir = -sign(e);
    this.dirAgreeLog(t,:) = (obsDir == desiredDQdir) & (obsDir ~= 0);

    % Métricas escalares
    this.meanDistStep = mean(abs(e));
    this.mseStep      = mean(e.^2);

    thrSuccess = 0.20;
    thrNear    = 0.30;

    absErr = abs(e);
    this.successStep     = all(absErr < thrSuccess);
    this.nearSuccessStep = all(absErr < thrNear);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (7) LOGS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    this.emgLog{this.c} = emg;
    this.encoderAdjustedLog{this.c} = this.adjustEnc;
    this.rewardLog(this.c) = reward;
    this.rewardIndividualLog{this.c} = rewardVector;
    this.flexConvertedLog{this.c} = this.flexConverted;

    this.meanDistLog(this.c) = this.meanDistStep;
    this.mseLog(this.c)      = this.mseStep;
    this.successLog(this.c)  = this.successStep;
    this.nearSuccessLog(this.c) = this.nearSuccessStep;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (8) TERMINAL CONDITION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    isDone = this.checkEndEpisode();

    if isDone
        this.episodeCount = this.episodeCount + 1;
        this.meanDistEpisode(this.episodeCount) = mean(this.meanDistLog(1:this.c), 'omitnan');
        this.successRateEpisode(this.episodeCount) = mean(this.successLog(1:this.c), 'omitnan');
        this.mseEpisode(this.episodeCount) = mean(this.mseLog(1:this.c), 'omitnan');

        % ===== métricas finales del episodio =====
        finalErr = abs(e);

        this.finalAbsErrEpisode = finalErr;
        this.finalMeanAbsErr    = mean(finalErr);
        this.finalMaxAbsErr     = max(finalErr);
        this.nearSuccessEpisode = any(this.nearSuccessLog(1:this.c) == 1);

        % ===== resumen de acciones usadas =====
        A_ep = this.aRawLog(1:this.c,:);
        [Auniq, ~, ic] = unique(A_ep, 'rows');
        counts = accumarray(ic, 1);

        fprintf("\n[EP DIAG] final abs error per motor = %s\n", mat2str(finalErr,4));
        fprintf("[EP DIAG] final mean(abs(err)) = %.6f | final max(abs(err)) = %.6f\n", ...
            this.finalMeanAbsErr, this.finalMaxAbsErr);
        fprintf("[EP DIAG] strict success ever = %d | near success ever = %d\n", ...
            any(this.successLog(1:this.c) == 1), this.nearSuccessEpisode);

        fprintf("[EP DIAG] unique actions used = %d\n", size(Auniq,1));
        for k = 1:size(Auniq,1)
            fprintf("  action %s -> %d times\n", mat2str(Auniq(k,:)), counts(k));
        end

        %-----------------------------------------------------------------
        encEp = this.encRawLog(1:this.c,:);
        encMinEp = min(encEp, [], 1);
        encMaxEp = max(encEp, [], 1);
        
        fprintf("[ENC RANGE] min per motor = %s\n", mat2str(encMinEp,4));
        fprintf("[ENC RANGE] max per motor = %s\n", mat2str(encMaxEp,4));
        %-----------------------------------------------------------------
        %

        %-----------------------------------------------------------------        
        % ===== NUEVO DIAGNOSTICO POR MOTOR =====
        dEncPerMotor = diff(this.encRawLog(1:this.c,:),1,1);
        meanAbsDEncPerMotor = mean(abs(dEncPerMotor),1,'omitnan');
        
        fprintf('[ENC MOTOR DIAG] mean abs dEnc per motor = %s\n', ...
            mat2str(meanAbsDEncPerMotor,4));
        %-----------------------------------------------------------------


        fprintf("\n[ENC RAW] mean ||dEnc|| = %.6f | dead-zone raw = %.2f%%\n", ...
            mean(this.encEffectNormLog(1:this.c), 'omitnan'), ...
            100*mean(this.encEffectNormLog(1:this.c) < 1e-9, 'omitnan'));
    end

    if isDone && this.flagSaveTraining
        if isprop(this,'saveEpisodes') && this.saveEpisodes
            this.saveEpisode();
        end
    end

    notifyEnvUpdated(this);
    loggedSignals = [];
    drawnow
end




% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%% para funciones de recompensa V0 - V9 %%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% function [observation, reward, isDone, loggedSignals] = step(this, action)
% 
%     % Increment step counter
%     this.c = this.c + 1;
% 
%     % RESET logs por episodio
%     if this.c == 1
%         this.meanDistLog = NaN(1, this.maxNumberStepsInEpisodes);
%         this.mseLog      = NaN(1, this.maxNumberStepsInEpisodes);
%         this.successLog  = NaN(1, this.maxNumberStepsInEpisodes);
%     end
% 
%     % --- action checks
%     if this.unifyActions
%         action = action * [1 1 1 1];
%     end
%     if iscell(action), action = cell2mat(action); end
%     if ~isnumeric(action), error('action must be numeric'); end
% 
%     expectedActionSize = size(this.actionLog,2);
%     if size(action,2) ~= expectedActionSize
%         error('Action size does not match actionLog size');
%     end
% 
%     % Log raw action
%     this.actionLog(this.c,:) = action;
% 
%     % -----------------------------------------
%     % (1) Acción que se va a ejecutar (SAT opcional)
%     % -----------------------------------------
%     % Aquí NO calculamos reward todavía.
%     % Solo obtenemos la acción "saturada" si aplica.
%     actionSat = action;
%     if this.rf_modify_actions
%         % OJO: tu reward_function actualmente devuelve reward también.
%         % Para no romper, llamamos pero IGNORAMOS reward ahora.
%         % Si tu reward modifica la acción, aquí obtenemos actionSat.
% 
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%%%%%% para funciones de recompensa V0 - V9 %%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         [~, ~, actionSat] = this.reward_function(this, action, []);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     end
%     this.actionSatLog(this.c,:) = actionSat;
% 
%     % Aplicar speeds para el controlador
%     actionApplied = actionSat .* this.speeds;
% 
%     % -----------------------------------------
%     % (2) Ejecutar acción y leer sensores
%     % -----------------------------------------
%     drawnow
%     completed = this.prosthesis.sendAllSpeed( ...
%         actionApplied(1), actionApplied(2), actionApplied(3), actionApplied(4));
%     assert(completed, 'ERROR during sending speed to controller')
% 
%     while this.periodTic.toc() < this.period
%         drawnow
%     end
%     if this.wait_in_step
%         while toc(this.period_realTic) < this.period
%             drawnow
%         end
%     end
% 
%     if this.usePrerecorded
%         t_elapsed = this.periodTic.elapsed_time;
%         emg = this.myo.readEmg(t_elapsed);
%         flexData = this.glove.read(t_elapsed);
%     else
%         emg = this.myo.readEmg();
%         flexData = this.glove.read();
%     end
% 
%     motorData = this.prosthesis.read();
%     this.encoderLog{this.c} = motorData;
% 
%     % fallbacks
%     if isempty(emg), emg = this.emg; else, this.emg = emg; end
%     if isempty(motorData), motorData = this.motorData; else, this.motorData = motorData; end
%     if isempty(flexData), flexData = this.flexData; else, this.flexData = flexData; end
% 
%     % reset timers
%     this.periodTic.tic();
%     if this.wait_in_step, this.period_realTic = tic; end
% 
%     % -----------------------------------------
%     % (3) Actualizar estado y variables auxiliares
%     % -----------------------------------------
%     this.State = this.calculateState(emg, motorData);
%     observation = this.State;
% 
%     this.flexConverted = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
%     this.adjustEnc     = this.flexJoined_scaler(encoder2Flex(this.motorData));
% 
%     % -----------------------------------------
%     % (4) Ahora SÍ calcular reward (ya existe q_ref)
%     % -----------------------------------------
%     reward = 0;
%     rewardVector = zeros(1, expectedActionSize);
% 
%     if this.rf_modify_actions
%         % si tu reward depende de flexConverted/adjustEnc ya está listo
%         [reward, rewardVector, ~] = this.reward_function(this, actionSat, []);
%     else
%         [reward, rewardVector] = this.reward_function(this, actionSat, []);
%     end
% 
%     % -----------------------------------------
%     % (5) Métricas SIEMPRE en step.m
%     % -----------------------------------------
%     q     = this.adjustEnc(end,:);
%     q_ref = this.flexConverted(end,:);
%     e = q - q_ref;
% 
%     this.meanDistStep = mean(abs(e));
%     this.mseStep      = mean(e.^2);
% 
%     thrSuccess = 0.03;
%     this.successStep  = all(abs(e) < thrSuccess);
% 
%     % logs
%     this.rewardLog(this.c) = reward;
%     this.rewardIndividualLog{this.c} = rewardVector;
%     this.flexConvertedLog{this.c} = this.flexConverted;
%     this.encoderAdjustedLog{this.c} = this.adjustEnc;
%     this.emgLog{this.c} = emg;
% 
%     this.meanDistLog(this.c) = this.meanDistStep;
%     this.mseLog(this.c)      = this.mseStep;
%     this.successLog(this.c)  = this.successStep;
% 
%     % terminar episodio
%     isDone = this.checkEndEpisode();
% 
%     if isDone
%         this.episodeCount = this.episodeCount + 1;
%         this.meanDistEpisode(this.episodeCount) = mean(this.meanDistLog(1:this.c), 'omitnan');
%         this.successRateEpisode(this.episodeCount) = mean(this.successLog(1:this.c), 'omitnan');
%         this.mseEpisode(this.episodeCount) = mean(this.mseLog(1:this.c), 'omitnan');
%     end
% 
%     if isDone && this.flagSaveTraining
%         this.saveEpisode();
%     end
% 
%     notifyEnvUpdated(this);
%     loggedSignals = [];
%     drawnow
% end

















% function [observation, reward, isDone, loggedSignals] = step(this, action)
%     this.c = this.c + 1;
% 
%     %%%%%%%%%%%%%%%%%%%%------------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%------------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % ---- RESET por episodio: inicializar logs por step
%     if this.c == 1
%         this.meanDistLog = zeros(this.maxNumberStepsInEpisodes, 1);
%         this.mseLog      = zeros(this.maxNumberStepsInEpisodes, 1);
%         this.successLog  = zeros(this.maxNumberStepsInEpisodes, 1);
%     end
%     %%%%%%%%%%%%%%%%%%%%------------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%------------------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     %--- action verification depending matlab vers.
%     % if ~isequal(matlabRelease.Release, "R2021b")
%     %     %adjusting changes in toolbox apart from R2021b
%     %     action = action{1};
%     % end
% 
%     if this.unifyActions
%         action = action * [1 1 1 1];
%     end
% 
%     % Ensure action is of type double and has the correct size
%     if iscell(action)
%         action = cell2mat(action);
%     end
% 
%     if ~isnumeric(action)
%         error('action must be of numeric type');
%     end
% 
%     % Ensure action has the correct dimensions
%     expectedActionSize = size(this.actionLog, 2);
%     if size(action, 2) ~= expectedActionSize
%         error('The action size does not match the actionLog size');
%     end
% 
%     this.actionLog(this.c, :) = action;
%     fprintf('  actions=[%2d %2d %2d %2d]\n', action);
% 
%     if this.rf_modify_actions
%         % action is clipped
%         [reward, rewardVector, action] = this.reward_function(this, action, []);
%         fprintf('actionSat=[%2d %2d %2d %2d]\n', action);
%     %else
%         % calculate reward without modifying actions
%     %    [reward, rewardVector] = this.reward_function(this, action);
%     end
% 
%     this.actionSatLog(this.c, :) = action;
%     action = action .* this.speeds;
% 
%     %% applying action
%     drawnow
%     completed = this.prosthesis.sendAllSpeed(...
%         action(1), action(2), action(3), action(4));
% 
%     assert(completed, 'ERROR during sending speed to controller')
% 
%     %% waiting data, applying action.
%     while this.periodTic.toc() < this.period
%         drawnow
%     end
% 
%     if this.wait_in_step
%         % waiting when half-hardware execution
%         while toc(this.period_realTic) < this.period
%             drawnow
%         end
%     end
% 
%     %% reading hardware
%     if this.usePrerecorded
%         % only waits a period
%         t_elapsed = this.periodTic.elapsed_time;
%         % supposedly it is a period
%         assert(t_elapsed > 0.9*this.period && t_elapsed < 1.1*this.period, ...
%             "time elapsed %.2f is incorrect, must be %.2f", ...
%             t_elapsed, this.period)
%         emg = this.myo.readEmg(t_elapsed);
%         flexData = this.glove.read(t_elapsed);
%     else
%         emg = this.myo.readEmg(); % E-by-8
%         flexData = this.glove.read(); % n-by-9 double
%     end
% 
%     motorData = this.prosthesis.read(); % m-by-4 double
% 
%     this.encoderLog{this.c} = motorData;
% 
%     if isempty(emg)
%         emg = this.emg;
%         warning("--------------------emg is empty")
%     else
%         this.emg = emg;
%     end
% 
%     if isempty(motorData)
%         motorData = this.motorData;
%         warning("--------------------motorData is empty")
%     else
%         this.motorData = motorData;
%     end
% 
%     if isempty(flexData)
%         flexData = this.flexData;
%         warning("--------------------flexdata is empty")
%     else
%         this.flexData = flexData;
%     end
% 
%     % --- end step
%     this.periodTic.tic();
%     if this.wait_in_step
%         this.period_realTic = tic;
%     end
% 
%     this.log(sprintf(...
%         '%d. T=%.3f[s]. EmgSize %d. encodersSize %d. glovesize %d',...
%         this.c, this.episodeTic.toc(this.c),...
%         size(emg, 1), size(motorData, 1), size(flexData, 1)))
% 
%     %% Update prosthesis states
%     this.State = this.calculateState(emg, motorData);
%     observation = this.State;
% 
%     %% Reward
%     this.flexConverted = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
%     this.adjustEnc = this.flexJoined_scaler(encoder2Flex(this.motorData));
% 
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
%     %if this.rf_modify_actions
%         % action is not clipped
%     %    reward = this.reward_function(this, action, []);
%     %end
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     %% logs
%     this.emgLog{this.c} = emg;
%     this.encoderAdjustedLog{this.c} = this.adjustEnc;
%     this.rewardLog(this.c) = reward;
%     this.rewardIndividualLog{this.c} = rewardVector; % save individual rewards
%     this.flexConvertedLog{this.c} = this.flexConverted;
%     %%%%%%%%%%%%%%%%%%%------------------------------------------%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%------------------------------------------%%%%%%%%%%%%%%
%     % ---- Guardado de métricas por step ----
%     this.meanDistLog(this.c) = this.meanDistStep;
%     this.mseLog(this.c) = this.mseStep;
%     this.successLog(this.c) = this.successStep;
%     %%%%%%%%%%%%%%%%%%%------------------------------------------%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%------------------------------------------%%%%%%%%%%%%%%
% 
%     % disp('encoderAdjustedLog')
%     % disp(this.adjustEnc)
%     % disp('reward')
%     % disp(reward)
%     % disp('rewardVector')
%     % disp(rewardVector)
%     % disp('flexConvertedLog')
%     % disp(this.flexConverted)
% 
%     %% Check terminal condition
%     isDone = this.checkEndEpisode(); % || finishEpisode
% 
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
%     % ---- Si el episodio terminó, calcular métricas agregadas ----
%     % ---- Si el episodio terminó, calcular métricas agregadas ----
%     if isDone
%         % incrementar contador de episodios para métricas
%         this.episodeCount = this.episodeCount + 1;
% 
%         % promedios por episodio usando los steps realmente ejecutados (1:this.c)
%         this.meanDistEpisode(this.episodeCount) = mean(this.meanDistLog(1:this.c), 'omitnan');
%         this.successRateEpisode(this.episodeCount) = mean(this.successLog(1:this.c), 'omitnan');
%         this.mseEpisode(this.episodeCount) = mean(this.mseLog(1:this.c), 'omitnan');
%     end
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
%     if isDone && this.flagSaveTraining
%         this.saveEpisode();
%     end
%     % (optional) use notifyEnvUpdated to signal that the
%     % environment has been updated (e.g. to update visualization)
%     notifyEnvUpdated(this);
% 
%     loggedSignals = []; % not used
%     drawnow
% end
