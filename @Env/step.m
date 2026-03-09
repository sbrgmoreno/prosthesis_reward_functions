function [observation, reward, isDone, loggedSignals] = step(this, action)

    % Increment step counter
    this.c = this.c + 1;

    % ---- RESET por episodio: inicializar logs por step
    if this.c == 1
        this.meanDistLog = NaN(1, this.maxNumberStepsInEpisodes);
        this.mseLog      = NaN(1, this.maxNumberStepsInEpisodes);
        this.successLog  = NaN(1, this.maxNumberStepsInEpisodes);
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

    % Ensure action has correct dimensions
    expectedActionSize = size(this.actionLog, 2);
    if size(action,2) ~= expectedActionSize
        error('The action size does not match the actionLog size');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ===== DEBUG: fuerza acción para test de controlabilidad =====
    %action = ones(1,4);   % prueba también con -ones(1,4)
    if isprop(this,'forceActionDebug') && this.forceActionDebug
        action = this.forcedActionValue;
    end
    % ============================================================
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Log raw action (discrete)
    this.actionLog(this.c,:) = action;
    if this.verbose
        fprintf('  actions=[%2d %2d %2d %2d]\n', action);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (1) SATURACIÓN/CLIPPING DE ACCIÓN (SIN LLAMAR REWARD AQUÍ)
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fprintf("[DBG] actionSat=%s | speeds=%s | actionApplied=%s\n", ...
    mat2str(actionSat), mat2str(this.speeds), mat2str(actionApplied));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (2) APLICAR ACCIÓN
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow
    completed = this.prosthesis.sendAllSpeed( ...
        actionApplied(1), actionApplied(2), actionApplied(3), actionApplied(4));
    assert(completed, 'ERROR during sending speed to controller')

    % waiting data, applying action.
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    encRaw = motorData(end,:);   % 1x4 encoder crudo (última muestra)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    % --- end step (reset timers)
    this.periodTic.tic();
    if this.wait_in_step
        this.period_realTic = tic;
    end

    this.log(sprintf( ...
        '%d. T=%.3f[s]. EmgSize %d. encodersSize %d. glovesize %d', ...
        this.c, this.episodeTic.toc(this.c), ...
        size(emg, 1), size(motorData, 1), size(flexData, 1)));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (4) ACTUALIZAR ESTADO Y VARIABLES AUXILIARES (q y q_ref listos)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    this.State = this.calculateState(emg, motorData);
    observation = this.State;

    this.flexConverted = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
    % this.adjustEnc     = this.flexJoined_scaler(encoder2Flex(this.motorData));%*******||| ANTES DESCOMENTADO

    %%%%%%%%%%%%%%%%%%%%%%%%%    ^     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%    |     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%    |     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    enc = this.motorData(end,:);

    % Normalización temporal simple de encoder crudo
    encMin = [0 0 0 0];
    encMax = [5000 5000 5000 5000];   % ajusta si hace falta según tus rangos reales
    
    this.adjustEnc = max(0, min(1, (enc - encMin) ./ (encMax - encMin)));
    %%%%%%%%%%%%%%%%%%%%%% 1 LINEA COMENTADA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (5) REWARD (UNA SOLA VEZ, DESPUÉS DE TENER q y q_ref)  ✅ V10 OK
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    reward = 0;
    rewardVector = zeros(1, expectedActionSize);

    try
        [reward, rewardVector, ~] = this.reward_function(this, actionSat, []);
    catch
        [reward, rewardVector] = this.reward_function(this, actionSat, []);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (6) MÉTRICAS POR STEP (siempre aquí, NO dentro del reward)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    q     = this.adjustEnc(end,:);
    q_ref = this.flexConverted(end,:);
    e = q - q_ref;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % =========================
    % Sensitivity logging (C)
    % =========================
    t = this.c;
    
    % Init episode logs
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ============== NUEVO: logs de encoder crudo ================
        this.encRawLog        = NaN(this.maxNumberStepsInEpisodes, 4);
        this.encEffectNormLog = NaN(this.maxNumberStepsInEpisodes, 1);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    % Log q and q_ref
    this.qLog(t,:)    = q;
    this.qRefLog(t,:) = q_ref;
    
    % Log actions
    this.aRawLog(t,:)     = actionSat;        % (-1,0,1)
    this.aAppliedLog(t,:) = actionApplied;    % after speeds
    
    % dq = q(t)-q(t-1)
    if t == 1
        dq = zeros(1,4);
    else
        dq = q - this.qLog(t-1,:);
    end
    this.dqLog(t,:) = dq;
    
    % effect size
    this.effectNormLog(t) = norm(dq);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % ===== NUEVO: efecto en encoder crudo =====
        this.encRawLog(t,:) = encRaw;
        
        if t == 1
            dEncRaw = zeros(1,4);
        else
            dEncRaw = encRaw - this.encRawLog(t-1,:);
        end
        
        this.encEffectNormLog(t) = norm(dEncRaw);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % error norm
    err = q - q_ref;
    this.errNormLog(t) = norm(err);
    
    % dErr = err(t-1)-err(t) (positive = improvement)
    if t == 1
        this.dErrLog(t) = 0;
    else
        this.dErrLog(t) = this.errNormLog(t-1) - this.errNormLog(t);
    end
    
    % Direction agreement: action pushes toward reducing error
    % If err(i)>0 => want negative dq (decrease q), so action should be -1
    % If err(i)<0 => want positive dq, so action should be +1



    % desiredDir = -sign(err);                % direction that would reduce error
    % this.dirAgreeLog(t,:) = (sign(actionSat) == desiredDir) & (actionSat ~= 0);
    
    %%%%%%%%%%%%%%%%%%    ^     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%    |     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%    |     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dirección real observada en q
    obsDir = sign(dq);             % dq = q(t)-q(t-1)
    
    % Dirección deseada para reducir error:
    % si err>0 queremos dq<0; si err<0 queremos dq>0
    desiredDQdir = -sign(err);
    
    % Agreement real: el dq observado va en la dirección deseada
    this.dirAgreeLog(t,:) = (obsDir == desiredDQdir) & (obsDir ~= 0);
    %%%%%%%%%%%%%%% 2 LINEAS COMENTADAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    this.meanDistStep = mean(abs(e));
    this.mseStep      = mean(e.^2);

    thrSuccess = 0.15; %0.03;
    this.successStep  = all(abs(e) < thrSuccess);

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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % (8) TERMINAL CONDITION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    isDone = this.checkEndEpisode();

    if isDone
        this.episodeCount = this.episodeCount + 1;
        this.meanDistEpisode(this.episodeCount) = mean(this.meanDistLog(1:this.c), 'omitnan');
        this.successRateEpisode(this.episodeCount) = mean(this.successLog(1:this.c), 'omitnan');
        this.mseEpisode(this.episodeCount) = mean(this.mseLog(1:this.c), 'omitnan');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        fprintf("\n[ENC RAW] mean ||dEnc|| = %.6f | dead-zone raw = %.2f%%\n", ...
        mean(this.encEffectNormLog(1:this.c), 'omitnan'), ...
        100*mean(this.encEffectNormLog(1:this.c) < 1e-9, 'omitnan'));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    end

    if isDone && this.flagSaveTraining
        %this.saveEpisode(); %antes descomentado 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if isprop(this,'saveEpisodes') && this.saveEpisodes
            this.saveEpisode();
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
