function [reward, rewardVector, action] = legacy_distanceRewarding(this, action)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      VERSION v0     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [reward, rewardVector, action] = reward_v0_tracking(this, action)
% ===========================
% Reward v0: Tracking puro (baseline)
% - Sin PBRS
% - Sin progreso
% - Sin bonos
% - Sin penalizaciones heurísticas por dirección/suavidad/inactividad
% - Solo: minimizar error entre posFlex y flexConv(t)
% ===========================

% % ---- (1) Obtener referencia del dataset en este step: flexConv
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% 
% % ---- (2) Obtener estado actual de motores y convertir a flex normalizada: posFlex
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% % ---- (3) Error por motor (tracking): diff = posFlex - target
% diff = posFlex - flexConv(end, :);      % 1x4
% 
% % ---- (4) Métrica escalar del error (baseline): MSE (mean squared error)
% mse = mean(diff.^2);                    % escalar
% %%%%%%%%%%%%%%%%%%%------------------------------------------%%%%%%%%%%%%%%
% %--------------------------------------------------------------------------
% meanAbsDist = mean(abs(diff));  % más interpretable que mse
% %--------------------------------------------------------------------------
% %%%%%%%%%%%%%%%%%%%------------------------------------------%%%%%%%%%%%%%%
% 
% % ---- (5) Reward base: negativo del error (minimizar mse)
% k_d = 1.0;                              % escala inicial (igual que opts.k en v5)
% rewardRaw = -k_d * mse;                 % escalar
% 
% % ---- (6) Sin clipping en v0 (para ver comportamiento real del error)
% reward = rewardRaw;
% 
% % ---- (7) rewardVector (para compatibilidad): descomposición por motor
% % Aquí lo hacemos como contribución negativa por motor (opcional, pero útil)
% rewardVector = -k_d * (diff.^2);        % 1x4
% 
% % ---- (8) DEBUG (opcional)
% if this.c == 1
%     fprintf("---- NEW EP (v0) ----\n");
% end
% if mod(this.c,10)==0
%     distanceAbs = abs(diff);
%     meanAbsDist = mean(distanceAbs);
%     fprintf("c=%d reward=%.6f mse=%.6f | meanAbsDist=%.4f | minAbsDist=%.4f\n", ...
%         this.c, reward, mse, meanAbsDist, min(distanceAbs));
% end
% 
% % Guardar métricas en el entorno
% this.meanDistStep = meanAbsDist;
% this.mseStep = mse;
% this.successStep = all(abs(diff) < 0.03);
% 
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      VERSION v0.1   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%function [reward, rewardVector, action] = reward_v01_tracking_progress(this, action, ~)
% ===========================
% Reward v0.1: Tracking + Progreso
% - Base: -k_d * mse
% - Progreso: k_p * (mse_prev - mse) con clamp
% - Sin PBRS, sin bonuses, sin tanh
% - Registra métricas: meanDistStep, mseStep, successStep
% ===========================

% persistent prevMSE
% 
% % ---- RESET por episodio
% % this.c es el contador de steps dentro del episodio (tu Env lo usa así)
% if this.c == 1
%     prevMSE = NaN;   % evita castigo artificial en el primer step
% end
% 
% % ---- (1) Referencia del dataset (q_ref) en este step
% % Nota: en tu implementación, flexConv(end,:) es el target de los 4 motores
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% 
% q_ref = flexConv(end, :);   % 1x4
% 
% % ---- (2) Estado actual (q) desde encoders
% pos = this.motorData(end, :);
% q   = this.flexJoined_scaler(encoder2Flex(pos));   % 1x4
% 
% % ---- (3) Error y métricas
% diff = q - q_ref;                % 1x4
% mse  = mean(diff.^2);            % escalar (tracking loss)
% meanAbsDist = mean(abs(diff));   % escalar (interpretación directa)
% successStep = all(abs(diff) < 0.03);
% 
% % ---- Guardar métricas en el entorno (para logs en step.m)
% this.meanDistStep = meanAbsDist;
% this.mseStep      = mse;
% this.successStep  = successStep;
% 
% % ---- (4) Término base (tracking)
% k_d = 1.0;
% baseTerm = -k_d * mse;
% 
% % ---- (5) Progreso temporal
% k_p = 80;  % punto de partida (si hay mucha varianza baja a 40-60)
% if isnan(prevMSE)
%     progressTerm = 0;
% else
%     progressTerm = k_p * (prevMSE - mse);
% end
% prevMSE = mse;
% 
% % Clamp del progreso para estabilidad (muy importante)
% progressTerm = max(min(progressTerm, 5), -5);
% 
% % ---- (6) Reward total
% rewardRaw = baseTerm + progressTerm;
% reward = rewardRaw;
% 
% % ---- (7) rewardVector (por motor) para compatibilidad/diagnóstico
% % contribución negativa por motor (error cuadrático por articulación)
% rewardVector = -k_d * (diff.^2);
% 
% % ---- Debug opcional (cada 10 steps)
% % if this.c == 1
% %     fprintf("---- NEW EP (v0.1) ----\n");
% % end
% % if mod(this.c,10)==0
% %     fprintf("c=%d base=%.6f prog=%.3f total=%.6f | mse=%.6f | meanAbsDist=%.4f | succ=%d\n", ...
% %         this.c, baseTerm, progressTerm, reward, mse, meanAbsDist, successStep);
% % end
% 
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% VERSION 1 PBRS + CLIPPING  %%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%function [reward, rewardVector, action] = reward_v1_pbrs_minimal(this, action, ~)
% ==========================================================
% Reward v1 (Minimal): Tracking + PBRS + Smooth Clipping
% ----------------------------------------------------------
% r = -k_d * mse  +  k_s * (gamma*Phi(s') - Phi(s))
% Phi(s) = -log(1 + mse)
% reward = L * tanh(r_raw / L)
%
% - Sin bonuses heurísticos (precisión/estabilidad/etc.)
% - Sin término de progreso explícito (ya va implícito en PBRS)
% - Registra métricas: meanDistStep, mseStep, successStep
% ==========================================================

% persistent prevPhi
% 
% % ---- RESET por episodio
% if this.c == 1
%     prevPhi = 0;
% end
% 
% % ---- (1) Referencia (q_ref) del dataset en este step
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% q_ref = flexConv(end, :);  % 1x4
% 
% % ---- (2) Estado actual (q) desde encoders
% pos = this.motorData(end, :);
% q   = this.flexJoined_scaler(encoder2Flex(pos));  % 1x4
% 
% % ---- (3) Error y métricas
% diff = q - q_ref;              % 1x4
% mse  = mean(diff.^2);          % escalar
% meanAbsDist = mean(abs(diff)); % escalar (más interpretable)
% successStep = all(abs(diff) < 0.03);
% 
% % ---- Guardar métricas en el entorno (para logs en step.m)
% this.meanDistStep = meanAbsDist;
% this.mseStep      = mse;
% this.successStep  = successStep;
% 
% % ---- (4) Parámetros (iniciales recomendados)
% k_d   = 1.0;    % peso del tracking (base)
% gamma = 0.99;   % para PBRS
% k_s   = 10;     % peso del shaping PBRS (sube a 20 si señal débil)
% L     = 10;     % límite de clipping suave (10 o 20)
% 
% % ---- (5) Término base (tracking)
% baseTerm = -k_d * mse;
% 
% % ---- (6) Potencial y PBRS
% phi = -log(1 + mse);                 % Phi(s')
% shapingTerm = k_s * (gamma*phi - prevPhi);
% prevPhi = phi;
% 
% % ---- (7) Reward total + clipping suave
% rewardRaw = baseTerm + shapingTerm;
% reward = L * tanh(rewardRaw / L);
% 
% % ---- (8) rewardVector (por motor) para compatibilidad/diagnóstico
% rewardVector = -k_d * (diff.^2);     % contribución negativa por articulación
% 
% % ---- Debug opcional (cada 10 steps)
% % if this.c == 1
% %     fprintf("---- NEW EP (v1 PBRS minimal) ----\n");
% % end
% % if mod(this.c,10)==0
% %     fprintf("c=%d base=%.6f shape=%.6f raw=%.6f final=%.6f | mse=%.6f | meanAbs=%.4f | succ=%d\n", ...
% %         this.c, baseTerm, shapingTerm, rewardRaw, reward, mse, meanAbsDist, successStep);
% % end
% 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%% VERSION 1.1 PBRS + CLIPPING + BONUS POR PREPOSICION    %%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% %function [reward, rewardVector, action] = reward_v11_pbrs_precision(this, action, ~)
% % ==========================================================
% % Reward v1.1: Stronger Tracking + PBRS + Soft Precision Bonus + Clipping
% % ----------------------------------------------------------
% % r = -k_d*mse + k_s*(gamma*Phi(s') - Phi(s)) + k_prec*exp(-alpha*mse)
% % Phi(s) = -log(1 + mse)
% % reward = L * tanh(r_raw / L)
% %
% % - Mantiene estabilidad (clipping)
% % - Aumenta presión por reducir error (k_d alto)
% % - Añade empuje suave hacia precisión fina (bonus continuo)
% % - Registra métricas: meanDistStep, mseStep, successStep
% % ==========================================================
% 
% persistent prevPhi
% 
% % ---- RESET por episodio
% if this.c == 1
%     prevPhi = 0;
% end
% 
% % ---- (1) Referencia (q_ref) del dataset en este step
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% q_ref = flexConv(end, :);  % 1x4
% 
% % ---- (2) Estado actual (q) desde encoders
% pos = this.motorData(end, :);
% q   = this.flexJoined_scaler(encoder2Flex(pos));  % 1x4
% 
% % ---- (3) Error y métricas
% diff = q - q_ref;              % 1x4
% mse  = mean(diff.^2);          % escalar
% meanAbsDist = mean(abs(diff)); % escalar
% successStep = all(abs(diff) < 0.03);
% 
% % ---- Guardar métricas en el entorno (para logs en step.m)
% this.meanDistStep = meanAbsDist;
% this.mseStep      = mse;
% this.successStep  = successStep;
% 
% % ---- (4) Parámetros (v1.1 recomendados)
% k_d   = 8.0;     % ↑ más presión por tracking (prueba 5..12)
% gamma = 0.99;    % PBRS
% k_s   = 10.0;    % shaping PBRS (mantener para estabilidad)
% 
% % Bonus suave de precisión (empuja refinamiento)
% k_prec = 1.5;    % prueba 1..3
% alpha  = 60;     % sensibilidad (30..120). Mayor => bonus solo cuando mse es pequeño
% 
% % Clipping suave
% L = 12;          % 10..20 (sube si sientes mucha saturación)
% 
% % ---- (5) Término base (tracking fuerte)
% baseTerm = -k_d * mse;
% 
% % ---- (6) Potencial y PBRS
% phi = -log(1 + mse);                 % Phi(s')
% shapingTerm = k_s * (gamma*phi - prevPhi);
% prevPhi = phi;
% 
% % ---- (7) Bonus suave de precisión
% precisionBonus = k_prec * exp(-alpha * mse);
% 
% % ---- (8) Reward total + clipping
% rewardRaw = baseTerm + shapingTerm + precisionBonus;
% reward = L * tanh(rewardRaw / L);
% 
% % ---- (9) rewardVector (por motor) para compatibilidad/diagnóstico
% rewardVector = -k_d * (diff.^2);
% 
% % ---- Debug opcional
% % if this.c == 1
% %     fprintf("---- NEW EP (v1.1 PBRS+precision) ----\n");
% % end
% % if mod(this.c,10)==0
% %     fprintf("c=%d base=%.6f shape=%.6f prec=%.6f raw=%.6f final=%.6f | mse=%.6f | meanAbs=%.4f | succ=%d\n", ...
% %         this.c, baseTerm, shapingTerm, precisionBonus, rewardRaw, reward, mse, meanAbsDist, successStep);
% % end
% 
% end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%% %%      VERSION 2   tipo delta-error dominante + clipping%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% 
% %function [reward, rewardVector, action] = reward_v2_progress_dominant(this, action, ~)
% % ==========================================================
% % Reward v2: Progress-Dominant (Tracking + Delta-Error + Clipping)
% % ----------------------------------------------------------
% % r = -k_d*mse + k_p*(mse_prev - mse) - k_bad*max(0, mse - mse_prev)
% % reward = L * tanh(r_raw / L)
% %
% % Objetivo: forzar mejora step-a-step (progreso explícito).
% % ==========================================================
% 
% persistent prevMSE
% 
% % ---- RESET por episodio
% if this.c == 1
%     prevMSE = NaN; % evita castigo artificial al primer step
% end
% 
% % ---- (1) Referencia (q_ref) del dataset en este step
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% q_ref = flexConv(end, :);  % 1x4
% 
% % ---- (2) Estado actual (q) desde encoders
% pos = this.motorData(end, :);
% q   = this.flexJoined_scaler(encoder2Flex(pos));  % 1x4
% 
% % ---- (3) Error y métricas
% diff = q - q_ref;              % 1x4
% mse  = mean(diff.^2);          % escalar
% meanAbsDist = mean(abs(diff)); % escalar
% successStep = all(abs(diff) < 0.03);
% 
% % ---- Guardar métricas en el entorno (para logs en step.m)
% this.meanDistStep = meanAbsDist;
% this.mseStep      = mse;
% this.successStep  = successStep;
% 
% % ---- (4) Parámetros (iniciales recomendados)
% k_d   = 1.0;      % tracking base (suave)
% k_p   = 250.0;    % progreso dominante (prueba 150..400)
% k_bad = 120.0;    % castigo si empeora (prueba 60..200)
% 
% L = 12;           % clipping suave (10..20)
% 
% % ---- (5) Términos de reward
% baseTerm = -k_d * mse;
% 
% if isnan(prevMSE)
%     delta = 0;
% else
%     delta = (prevMSE - mse);   % positivo si mejora
% end
% 
% progressTerm = k_p * delta;
% 
% % Penaliza cuando empeora
% if isnan(prevMSE)
%     worsen = 0;
% else
%     worsen = max(0, mse - prevMSE);
% end
% badTerm = -k_bad * worsen;
% 
% % Actualizar memoria
% prevMSE = mse;
% 
% % ---- (6) Reward total + clipping
% rewardRaw = baseTerm + progressTerm + badTerm;
% reward = L * tanh(rewardRaw / L);
% 
% % ---- (7) rewardVector (por motor) para compatibilidad/diagnóstico
% % (solo componente base por articulación)
% rewardVector = -k_d * (diff.^2);
% 
% % ---- Debug opcional
% % if this.c == 1
% %     fprintf("---- NEW EP (v2 Progress Dominant) ----\n");
% % end
% % if mod(this.c,10)==0
% %     fprintf("c=%d base=%.6f prog=%.6f bad=%.6f raw=%.6f final=%.6f | mse=%.6f | meanAbs=%.4f\n", ...
% %         this.c, baseTerm, progressTerm, badTerm, rewardRaw, reward, mse, meanAbsDist);
% % end
% 
% end
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


% 
% persistent previousPosFlex inactiveSteps
% 
% if isempty(previousPosFlex)
%     previousPosFlex = zeros(size(action)); % Inicializa el registro de posición
% end
% if isempty(inactiveSteps)
%     inactiveSteps = 0; % Inicializa el contador de inactividad
% end
% 
% %% Configuración de recompensas
% opts.k = 3; % Penalización suavizada por distancia
% rewards.dirInverse = -5; % Penalización por moverse en dirección incorrecta
% rewards.wrongStop = -15; % Penalización por detenerse incorrectamente
% rewards.goodMove = 15; % Recompensa por moverse en la dirección correcta
% rewards.goodMove2 = 1;
% rewards.inactivityPenalty = -2; % Penalización base por inactividad
% rewards.moveIncentive = 5; % Incentivo por moverse
% rewards.precisionBonus = 10; % Bonificación por precisión
% rewards.smoothnessPenalty = -3; % Penaliza cambios bruscos
% rewards.efficiencyBonus = 3; % Bonificación por movimientos suaves
% 
% rewardVector = zeros(1, 4);
% 
% %% Lectura del estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% % disp('posision motor')
% % disp(pos)
% % disp('previousPosFlex')
% % disp(previousPosFlex)
% % disp('posFlex')
% % disp(posFlex)
% % disp('flexConv')
% % disp(flexConv)
% %% Evaluación de recompensa por cada motor
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;  % Mover hacia adelante
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1; % Mover hacia atrás
%     else
%         correctAction = 0;  % Mantenerse en su lugar
%     end
% 
%     % Aplicar recompensas y penalizaciones
%     if action(i) == correctAction
%         if action(i) ~= 0
%             rewardVector(i) = rewards.goodMove;
%         else
%             rewardVector(i) = rewards.goodMove2;
%         end
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     % Calcular la pendiente del movimiento
%     slope = (posFlex(i) - previousPosFlex(i));
% 
%     % Penalizar cambios bruscos con menor impacto
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
% 
%     % Bonificar movimientos eficientes con menor impacto
%     if abs(slope) > 0.01 && abs(slope) < 0.5
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% 
% % Actualizar el registro de posición
% previousPosFlex = posFlex;
% 
% %% Penalizacion acumulada por inactividad
% if all(action == 0) && correctAction ~= 0  % Si todas las acciones son cero (no movimiento)
%     inactiveSteps = inactiveSteps + 1; % Incrementar el contador de inactividad
%     penalty = rewards.inactivityPenalty * inactiveSteps; % Penalización acumulada
%     rewardVector = rewardVector + penalty; % Aplicar la penalización acumulada
% else
%     inactiveSteps = 0; % Reiniciar el contador de inactividad si se mueve
% end
% 
% % Incentivar movimiento si el agente no se queda inactivo
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% % Penalización más moderada por distancia usando raíz cuadrada
% distance = abs(posFlex - flexConv(end, :));
% rewardVector = rewardVector - sqrt(distance) .* opts.k;
% 
% % Bonificación suavizada si la distancia es menor a un umbral
% precisionMask = distance < 0.05; % Si la distancia es menor a 5% del rango
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus / 2;
% 
% % Calcular la recompensa total con menor varianza
% reward = mean(rewardVector);
% 
% end


%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% persistent previousPosFlex inactiveSteps previousPhi
% 
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% 
% opts.k = 3;
% rewards.dirInverse = -5;
% rewards.wrongStop = -15;
% rewards.goodMove = 15;
% rewards.goodMove2 = 1;
% rewards.inactivityPenalty = -2;
% rewards.moveIncentive = 5;
% rewards.precisionBonus = 10;
% rewards.smoothnessPenalty = -3;
% rewards.efficiencyBonus = 3;
% baseShapingGain = 6;
% gamma = 0.99;
% 
% rewardVector = zeros(1, 4);
% 
% % Estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% % Recompensas motor por motor
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
% 
%     if abs(slope) > 0.01 && abs(slope) < 0.5
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% 
% previousPosFlex = posFlex;
% 
% % Penalización por inactividad
% if all(action == 0) && correctAction ~= 0
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% distance = abs(posFlex - flexConv(end, :));
% rewardVector = rewardVector - sqrt(distance) .* opts.k;
% 
% precisionMask = distance < 0.05;
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus / 2;
% 
% baseReward = mean(rewardVector);
% 
% %% 📐 Shaping: función de potencial y ajuste
% range = [4092 2046 1023 2046];  % rangos de normalización por motor
% normDiff = (posFlex - flexConv(end, :)) ./ range;
% phiCurrent = -log(1 + mean(normDiff .^ 2));  % función de potencial logarítmica
% 
% shapingTerm = baseShapingGain * (gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% reward = baseReward + shapingTerm;
% end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

% persistent previousPosFlex inactiveSteps previousPhi
% 
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% 
% %% Configuración
% opts.k = 3;
% gamma = 0.99;
% baseShapingGain = 6;
% clipLimit = 100;  % Límite para el clipping
% 
% % Recompensas
% rewards.dirInverse = -5;
% rewards.wrongStop = -15;
% rewards.goodMove = 15;
% rewards.goodMove2 = 1;
% rewards.inactivityPenalty = -2;
% rewards.moveIncentive = 5;
% rewards.precisionBonus = 10;
% rewards.smoothnessPenalty = -3;
% rewards.efficiencyBonus = 3;
% 
% rewardVector = zeros(1, 4);
% 
% %% Estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% %% Recompensas heurísticas motor por motor
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     % Suavidad
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
% 
%     % Eficiencia
%     if abs(slope) > 0.01 && abs(slope) < 0.5
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% 
% previousPosFlex = posFlex;
% 
% %% Penalización por inactividad
% if all(action == 0) && any(abs(posFlex - flexConv(end, :)) > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% % Incentivar movimiento
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% % Penalización por distancia
% distance = abs(posFlex - flexConv(end, :));
% rewardVector = rewardVector - sqrt(distance) .* opts.k;
% 
% % Bonificación por precisión
% precisionMask = distance < 0.05;
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus / 2;
% 
% %% Recompensa base
% baseReward = mean(rewardVector);
% 
% %% Shaping con función de potencial logarítmica
% range = [4092 2046 1023 2046];  % rangos de normalización por motor
% normDiff = (posFlex - flexConv(end, :)) ./ range;
% phiCurrent = -log(1 + mean(normDiff .^ 2));
% 
% shapingTerm = baseShapingGain * (gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% rewardRaw = baseReward + shapingTerm;
% 
% %% Clipping suave
% reward = clipLimit * tanh(rewardRaw / clipLimit);
% end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

% persistent previousPosFlex inactiveSteps previousPhi performanceWindow
% 
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% if isempty(performanceWindow), performanceWindow = zeros(1, 50); end
% 
% %% Parámetros
% opts.k = 3;
% gamma = 0.99;
% baseShapingGain = 6;
% clipLimit = 100;
% 
% rewards = struct( ...
%     "dirInverse", -5, ...
%     "wrongStop", -15, ...
%     "goodMove", 15, ...
%     "goodMove2", 1, ...
%     "inactivityPenalty", -2, ...
%     "moveIncentive", 5, ...
%     "precisionBonus", 10, ...
%     "smoothnessPenalty", -3, ...
%     "efficiencyBonus", 3, ...
%     "energyPenaltyWeight", -1.5 ...
% );
% 
% rewardVector = zeros(1, 4);
% 
% %% Estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% %% Recompensas por acción motor por motor
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     % Suavidad y eficiencia
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
%     if abs(slope) > 0.01 && abs(slope) < 0.5
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% previousPosFlex = posFlex;
% 
% %% Penalización por inactividad
% if all(action == 0) && any(abs(posFlex - flexConv(end, :)) > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% %% Incentivo por moverse
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% %% Penalización por distancia
% distance = abs(posFlex - flexConv(end, :));
% scaledDist = sqrt(distance);
% rewardVector = rewardVector - scaledDist .* opts.k;
% 
% %% Bonificación por precisión (Gauss adaptativa)
% sigma = max(0.01, mean(distance) * 0.5);  % Adaptar la sigma a la dificultad actual
% precisionBonus = rewards.precisionBonus * exp(-(distance.^2) / (2 * sigma^2));
% rewardVector = rewardVector + precisionBonus;
% 
% %% Penalización por energía
% energyPenalty = rewards.energyPenaltyWeight * mean(abs(action));
% rewardVector = rewardVector + energyPenalty;
% 
% %% Recompensa base
% baseReward = mean(rewardVector);
% 
% %% Reward shaping
% range = [4092 2046 1023 2046];
% normDiff = (posFlex - flexConv(end, :)) ./ range;
% phiCurrent = -log(1 + mean(normDiff .^ 2));
% 
% % Shaping adaptativo con performance reciente
% progressDelta = mean(previousPhi - phiCurrent);  % puede ser negativo
% performanceWindow = [performanceWindow(2:end), progressDelta];
% progressFactor = min(1.5, max(0.1, mean(performanceWindow) * 50)); % controlado
% shapingGain = baseShapingGain * progressFactor;
% 
% % Shaping term
% shapingTerm = shapingGain * (gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% rewardRaw = baseReward + shapingTerm;
% 
% %% Clipping suave
% reward = clipLimit * tanh(rewardRaw / clipLimit);
% end
% --------------------------------------------------------------------------
% --------------------------------------------------------------------------
% 
% persistent previousPosFlex inactiveSteps previousPhi
% 
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% 
% %% Parámetros de recompensa
% opts.k = 3;  % penalización por distancia
% opts.gamma = 0.99;
% opts.shapingGain = 6;
% opts.clipLimit = 100;
% range = [4092 2046 1023 2046];  % rango para normalización de motores
% 
% rewards.dirInverse = -5;
% rewards.wrongStop = -15;
% rewards.goodMove = 15;
% rewards.goodMove2 = 1;
% rewards.inactivityPenalty = -2;
% rewards.moveIncentive = 5;
% rewards.precisionBonus = 10;
% rewards.smoothnessPenalty = -3;
% rewards.efficiencyBonus = 3;
% 
% rewardVector = zeros(1, length(action));
% 
% %% Estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% %% Recompensa heurística motor a motor
% for i = 1:length(action)
%     % Determinar dirección correcta
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     % Aplicar recompensa según dirección
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     % Suavidad y eficiencia
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
%     if abs(slope) > 0.01 && abs(slope) < 0.5
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% 
% previousPosFlex = posFlex;
% 
% %% Penalización acumulada por inactividad
% if all(action == 0) && any(abs(posFlex - flexConv(end, :)) > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% %% Incentivo por moverse
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% %% Penalización por distancia (raíz cuadrada)
% distance = abs(posFlex - flexConv(end, :));
% rewardVector = rewardVector - sqrt(distance) * opts.k;
% 
% %% Bonificación por precisión
% precisionMask = distance < 0.05;
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus / 2;
% 
% %% Recompensa base
% baseReward = mean(rewardVector);
% 
% %% Reward shaping: potencial cuadrático normalizado
% normalizedDiff = (posFlex - flexConv(end, :)) ./ range;
% phiCurrent = -mean(normalizedDiff .^ 2);
% shapingTerm = opts.shapingGain * (opts.gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% %% Suma total con shaping
% rewardRaw = baseReward + shapingTerm;
% 
% %% Clipping suave con tanh
% reward = opts.clipLimit * tanh(rewardRaw / opts.clipLimit);
% end
%--------------------------------------------------------------------------
%--------------------------------------------------------------------------
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% MEJOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% persistent previousPosFlex inactiveSteps previousPhi stepCount
% 
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% if isempty(stepCount), stepCount = 0; end
% 
% stepCount = stepCount + 1;
% 
% %Parámetros de recompensa
% opts.k = 2.5;                      % penalización de distancia
% opts.gamma = 0.99;                % factor de descuento
% opts.baseShapingGain = 6;        % ganancia base del shaping
% opts.clipLimit = 100;            % límite base de clipping
% range = [4092 2046 1023 2046];    % rango para normalizar errores
% 
% %Recompensas
% rewards = struct( ...
%     'dirInverse', -5, ...
%     'wrongStop', -15, ...
%     'goodMove', 15, ...
%     'goodMove2', 2, ...
%     'inactivityPenalty', -2, ...
%     'moveIncentive', 4, ...
%     'precisionBonus', 10, ...
%     'smoothnessPenalty', -2, ...
%     'efficiencyBonus', 3, ...
%     'stabilityBonus', 4 ...
% );
% 
% rewardVector = zeros(1, length(action));
% 
% %Estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% %Recompensas por motor
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
%     if abs(slope) > 0.01 && abs(slope) < 0.4
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% 
% previousPosFlex = posFlex;
% 
% %Penalización por inactividad
% if all(action == 0) && any(abs(posFlex - flexConv(end, :)) > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% %Bonificación por moverse
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% %Penalización por distancia (suavizada)
% distance = abs(posFlex - flexConv(end, :));
% rewardVector = rewardVector - sqrt(distance) * opts.k;
% 
% %Bonificación por precisión
% precisionMask = distance < 0.03;
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus;
% 
% %Bonificación por estabilidad (todos los motores en sincronía)
% if std(distance) < 0.01
%     rewardVector = rewardVector + rewards.stabilityBonus;
% end
% 
% %Recompensa base
% baseReward = mean(rewardVector);
% 
% %Shaping con función de potencial logarítmica
% normalizedDiff = (posFlex - flexConv(end, :)) ./ range;
% phiCurrent = -log(1 + mean(normalizedDiff .^ 2));
% 
% %Ganancia adaptativa que decae suavemente en el tiempo
% shapingGain = opts.baseShapingGain * exp(-0.0002 * stepCount);
% 
% shapingTerm = shapingGain * (opts.gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% %Reward shaping + Clipping dinámico
% rewardRaw = baseReward + shapingTerm;
% reward = opts.clipLimit * tanh(rewardRaw / opts.clipLimit);
% 
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%VERSION MEJORADA NUEVA
%%%%%%RECOMPENSA%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% persistent previousPosFlex inactiveSteps previousPhi
% 
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% 
% % Parámetros generales
% opts.k = 3;                    % Penalización por distancia
% opts.gamma = 0.99;
% opts.shapingGain = 6;
% opts.clipLimit = 100;
% opts.oscPenalty = 1.5;         % Penalización por oscilaciones
% range = [4092 2046 1023 2046]; % Rango de normalización por motor
% weights = [0.3 0.25 0.25 0.2]; % Pesos por motor
% 
% % Recompensas
% rewards = struct( ...
%     'dirInverse', -5, ...
%     'wrongStop', -15, ...
%     'goodMove', 15, ...
%     'goodMove2', 1, ...
%     'inactivityPenalty', -2, ...
%     'moveIncentive', 5, ...
%     'precisionBonus', 10, ...
%     'smoothnessPenalty', -3, ...
%     'efficiencyBonus', 3, ...
%     'finalPrecisionBonus', 15 ...
% );
% 
% rewardVector = zeros(1, length(action));
% 
% % Obtener estado actual
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% % Recompensa heurística motor a motor
% directionChanges = 0;
% for i = 1:length(action)
%     % Dirección correcta
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     % Recompensa base
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     % Suavidad y eficiencia
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
%     if abs(slope) > 0.01 && abs(slope) < 0.5
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% 
%     % Conteo de cambios de dirección
%     if sign(slope) ~= sign(previousPosFlex(i) - flexConv(end, i))
%         directionChanges = directionChanges + 1;
%     end
% end
% 
% previousPosFlex = posFlex;
% 
% % Penalización por inactividad
% if all(action == 0) && any(abs(posFlex - flexConv(end, :)) > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     rewardVector = rewardVector + rewards.inactivityPenalty * inactiveSteps;
% else
%     inactiveSteps = 0;
% end
% 
% % Incentivo por moverse
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% % Penalización por distancia
% distance = abs(posFlex - flexConv(end, :));
% rewardVector = rewardVector - sqrt(distance) * opts.k;
% 
% % Bonificación por precisión parcial
% precisionMask = distance < 0.05;
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus / 2;
% 
% % Bonificación adicional si el error promedio final es muy bajo
% if mean(distance) < 0.03
%     rewardVector = rewardVector + rewards.finalPrecisionBonus;
% end
% 
% % Recompensa base total
% baseReward = mean(rewardVector);
% 
% % Función de potencial logarítmica ponderada
% normDiff = (posFlex - flexConv(end, :)) ./ range;
% phiCurrent = -log(1 + sum(weights .* (normDiff .^ 2)));
% 
% % Shaping con delta de potencial
% shapingTerm = opts.shapingGain * (opts.gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% % Penalización por oscilaciones
% oscPenalty = -opts.oscPenalty * directionChanges;
% 
% % Total sin clipping
% rewardRaw = baseReward + shapingTerm + oscPenalty;
% 
% % Clipping condicional
% if abs(rewardRaw) < opts.clipLimit
%     reward = rewardRaw;
% else
%     reward = opts.clipLimit * sign(rewardRaw);
% end
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     function [reward, rewardVector, action] = legacy_distanceRewarding(this, action)
% % =========================================================================
% % V6: Phase-aware PBRS + Soft Saturation  (diseñada para superar V5)
% % Firma original: [reward, rewardVector, action] = legacy_distanceRewarding(this, action)
% % =========================================================================
% 
%     % -------- Persistentes (como en tus versiones) --------
%     persistent previousPosFlex previousPhi prevDirSign stallCount
%     if isempty(previousPhi), previousPhi = 0; end
%     if isempty(stallCount),  stallCount  = 0; end
% 
%     % -------- 1) Leer estado/objetivo desde "this" --------
%     % Ajusta los nombres si en tu clase son diferentes.
%     posFlex  = getFirstProp(this, {'posFlex','PosFlex','currentPosFlex','flexState','stateFlex'});
%     flexConv = getFirstProp(this, {'flexConv','FlexConv','targetFlex','goalFlex','refFlex'});
% 
%     posFlex  = posFlex(:);
%     flexConv = flexConv(:);
%     n = numel(posFlex);
% 
%     if isempty(previousPosFlex), previousPosFlex = posFlex; end
%     if isempty(prevDirSign),     prevDirSign     = zeros(n,1); end
% 
%     % -------- 2) Parámetros (con defaults seguros) --------
%     gamma = getFirstProp(this, {'gamma','Gamma','discountFactor','DiscountFactor'}, 0.99);
% 
%     % Pesos por actuador (si ya tienes weights en tu clase, se usan)
%     w = getFirstProp(this, {'weights','w','motorWeights','W'}, ones(n,1));
%     w = w(:);
%     if numel(w) ~= n, w = ones(n,1); end
% 
%     % Puedes tener struct opts/rewards en tu clase; si existe se usa
%     opts = getFirstProp(this, {'opts','rewardOpts','rewardOptions'}, struct());
%     rewards = getFirstProp(this, {'rewards','rewardParams'}, struct()); %#ok<NASGU>
% 
%     % Defaults (ajusta según tu escala 0..1 o grados)
%     huberDelta     = getFieldOr(opts,'huberDelta',     0.05);  % robustez
%     stabilityEps   = getFieldOr(opts,'stabilityEps',   0.03);  % umbral de "ya llegué"
%     betaStability  = getFieldOr(opts,'betaStability',  0.50);  % cuánto vale sostener estable
%     lambdaOsc      = getFieldOr(opts,'lambdaOsc',      0.02);  % castigo oscilación
%     lambdaAct      = getFieldOr(opts,'lambdaAct',      0.01);  % castigo esfuerzo
%     stallTol       = getFieldOr(opts,'stallTol',       1e-3);  % mejora mínima
%     stallMax       = getFieldOr(opts,'stallMax',       8);     % pasos sin progreso
%     stallPenalty   = getFieldOr(opts,'stallPenalty',   0.05);  % penalización por estancamiento
%     stepCost       = getFieldOr(opts,'stepCost',       0.0);   % costo por paso (opcional)
%     Lsat           = getFieldOr(opts,'Lsat',           1.0);   % saturación suave
% 
%     % -------- 3) Reward "de tarea" (opcional) --------
%     % Si tu entorno tiene flags explícitos (success/fail), conecta aquí.
%     r_task = 0;
% 
%     % Ejemplo: "éxito" si todos los errores están bajo umbral
%     e = posFlex - flexConv;
%     isStableNow = all(abs(e) < stabilityEps);
%     if isStableNow
%         r_task = r_task + getFieldOr(opts,'R_success', 0.0);
%     end
% 
%     % -------- 4) Potencial Φ(s): progreso robusto + estabilidad --------
%     % Φ = -Σ w_i * Huber(e_i) + beta * I(estabilidad)
%     phiProgress = -sum(w .* huberLoss(e, huberDelta));
%     phiStab     = betaStability * double(isStableNow);
%     phiCurrent  = phiProgress + phiStab;
% 
%     % PBRS shaping
%     shapingTerm = gamma * phiCurrent - previousPhi;
% 
%     % -------- 5) Penalización por oscilación (chattering) --------
%     dir = sign(posFlex - previousPosFlex);                 % dirección real del cambio
%     dir(dir==0) = prevDirSign(dir==0);                     % evita ceros
%     directionChanges = sum(dir ~= prevDirSign);
%     p_osc = -lambdaOsc * directionChanges;
% 
%     % -------- 6) Penalización por esfuerzo --------
%     action = action(:);
%     p_act = -lambdaAct * sum(abs(action));
% 
%     % -------- 7) Penalización por estancamiento (stall) --------
%     prevErr = norm(previousPosFlex - flexConv, 1);
%     currErr = norm(posFlex - flexConv, 1);
%     improvement = prevErr - currErr;
% 
%     if improvement < stallTol
%         stallCount = stallCount + 1;
%     else
%         stallCount = max(stallCount - 1, 0);
%     end
% 
%     p_stall = 0;
%     if stallCount >= stallMax
%         p_stall = -stallPenalty;
%     end
% 
%     % -------- 8) Reward total (sin clipping duro) --------
%     r_raw = r_task + shapingTerm + p_osc + p_act + p_stall - stepCost;
% 
%     % Saturación suave (mejor que clipping para “superar V5”)
%     reward = Lsat * tanh( r_raw / max(Lsat, eps) );
% 
%     % -------- 9) rewardVector por actuador --------
%     % Distribuimos contribución por motor (útil para debug/plots).
%     % Base por motor: -w_i * huber(e_i)  (progreso)
%     % + un pequeño share del shaping global (para que sea interpretable)
%     rv_progress = -(w .* huberLoss(e, huberDelta));
%     rv_shapeShare = (shapingTerm / max(n,1)) * ones(n,1);
% 
%     % Penalizaciones globales repartidas (solo para lectura)
%     rv_penShare = ((p_osc + p_act + p_stall - stepCost) / max(n,1)) * ones(n,1);
% 
%     rewardVector = rv_progress + rv_shapeShare + rv_penShare;
% 
%     % -------- 10) Update persistentes --------
%     previousPosFlex = posFlex;
%     previousPhi     = phiCurrent;
%     prevDirSign     = dir;
% 
% end
% 
% % ================= Helpers =================
% 
% function val = getFirstProp(obj, names, default)
%     if nargin < 3, default = []; end
%     val = default;
%     for k = 1:numel(names)
%         nm = names{k};
%         try
%             if isprop(obj, nm)
%                 val = obj.(nm);
%                 if ~isempty(val), return; end
%             end
%         catch
%             % ignore y sigue
%         end
%         try
%             if isfield(obj, nm) %#ok<ISFLD>
%                 val = obj.(nm);
%                 if ~isempty(val), return; end
%             end
%         catch
%         end
%     end
% end
% 
% function v = getFieldOr(s, field, default)
%     v = default;
%     if isstruct(s) && isfield(s, field)
%         v = s.(field);
%     end
% end
% 
% function y = huberLoss(x, delta)
%     ax = abs(x);
%     y = zeros(size(x));
%     q = ax <= delta;
%     y(q)  = 0.5 * (x(q).^2);
%     y(~q) = delta * (ax(~q) - 0.5*delta);
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FIX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % ===========================
% % Reward v5: Shaping + Clipping (FIXED)
% % ===========================
% persistent previousPosFlex inactiveSteps previousPhi stepCount
% persistent prevMeanDist
% 
% % Init (first time)
% if isempty(prevMeanDist), prevMeanDist = 0; end
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% if isempty(stepCount), stepCount = 0; end
% 
% % ---- RESET por episodio (CRÍTICO)
% % En tu Env, this.c es el contador de steps dentro del episodio.
% % Si this.c==1 estamos en el primer step del episodio.
% if this.c == 1
%     previousPosFlex = zeros(size(action));
%     inactiveSteps = 0;
%     previousPhi = 0;
%     stepCount = 0;
%     % NO inicialices prevMeanDist en 0 (eso mete castigo artificial)
%     prevMeanDist = NaN;
% end
% 
% stepCount = stepCount + 1;
% 
% % Parámetros
% opts.k = 1.0;                    % penalización de distancia (suavizada)
% opts.gamma = 0.99;               % factor de descuento (para PBRS)
% opts.baseShapingGain = 20;       % 6 ganancia base del shaping
% opts.clipLimit = 20;            % límite base de clipping (puedes bajar a 10 si quieres ver más dinámica)
% % Nota: removemos 'range' porque posFlex y flexConv ya están en [0,1]
% 
% % Recompensas (por motor y globales)
% rewards = struct( ...
%     'dirInverse', -3, ...
%     'wrongStop', -6, ...
%     'goodMove', 15, ...
%     'goodMove2', 2, ...
%     'inactivityPenalty', -2, ...
%     'moveIncentive', 4, ...
%     'precisionBonus', 3, ...
%     'smoothnessPenalty', -2, ...
%     'efficiencyBonus', 3, ...
%     'stabilityBonus', 2 ...
% );
% 
% rewardVector = zeros(1, length(action));
% 
% % Estado actual (referencia glove y estado prótesis)
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% 
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% % ---------------------------
% % Recompensas por motor
% % ---------------------------
% for i = 1:length(action)
%     % Determinar acción correcta según tracking hacia flexConv
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     % Recompensa por dirección
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     % Penalización por suavidad (cambio brusco vs step anterior)
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
% 
%     % Bonus de eficiencia (movimiento pequeño pero útil)
%     if abs(slope) > 0.01 && abs(slope) < 0.4
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% 
% previousPosFlex = posFlex;
% 
% % ---------------------------
% % Penalización por inactividad
% % ---------------------------
% if all(action == 0) && any(abs(posFlex - flexConv(end, :)) > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% % ---------------------------
% % Bonificación por moverse
% % ---------------------------
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% % ---------------------------
% % Penalización por distancia (suavizada con sqrt)
% % ---------------------------
% distance = abs(posFlex - flexConv(end, :));
% %-----------------------
% meanDist = mean(distance);
% 
% % Reward por progreso (si reduce el error => positivo)
% kProgress = 40;  % prueba 20..80
% if isnan(prevMeanDist)
%     progressTerm = 0;            % no penalizar el primer step
% else
%     progressTerm = kProgress * (prevMeanDist - meanDist);
% end
% 
% prevMeanDist = meanDist;
% 
% progressTerm = max(min(progressTerm, 5), -5);   % clamp en [-5, +5]
% %---------------------------
% 
% rewardVector = rewardVector - sqrt(distance) * opts.k;
% 
% 
% % ---------------------------
% % Bonificación por precisión
% % ---------------------------
% precisionMask = distance < 0.03;
% rewardVector(precisionMask) = rewardVector(precisionMask) + rewards.precisionBonus;
% 
% % ---------------------------
% % Bonificación por estabilidad (sincronía entre motores)
% % ---------------------------
% if std(distance) < 0.01
%     rewardVector = rewardVector + rewards.stabilityBonus;
% end
% 
% % Recompensa base (promedio por motor)
% baseReward = sum(rewardVector);   %mean(rewardVector);
% 
% % ---------------------------
% % Bonificación cuando esten cerca los motores
% % ---------------------------
% if all(distance < 0.03)
%     baseReward = baseReward + 60;   % bonus de éxito
% end
% 
% 
% % ---------------------------
% % PBRS: potencial (FIXED SCALE)
% % ---------------------------
% % Como posFlex y flexConv ya están normalizados en [0,1],
% % no dividas por rangos gigantes, porque aplastas el potencial a ~0.
% diff = (posFlex - flexConv(end, :));      % ~[-1,1]
% phiCurrent = -log(1 + mean(diff.^2));     % potencial informativo
% 
% % Ganancia adaptativa (con reset por episodio, ya no muere globalmente)
% shapingGain = opts.baseShapingGain * exp(-0.0002 * stepCount);
% 
% % Término shaping (PBRS)
% shapingTerm = shapingGain * (opts.gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% % Reward shaping + Clipping dinámico
% rewardRaw = baseReward + shapingTerm + progressTerm;
% reward = opts.clipLimit * tanh(rewardRaw / opts.clipLimit);
% 
% % ================= DEBUG BLOCK =================
% if this.c == 1
%     fprintf("---- NEW EP ----\n");
% end
% 
% if mod(this.c,10)==0
%     fprintf("c=%d base=%.3f shape=%.3f raw=%.3f final=%.3f | meanDist=%.3f | minDist=%.3f\n", ...
%     this.c, baseReward, shapingTerm, rewardRaw, reward, meanDist, min(distance));
% 
% end
% % ===============================================
% 
% end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%% VERSION 7 v5 + v6 (PBRS + clipping + Huber loss)  %%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %function [reward, rewardVector, action] = legacy_distanceRewarding_v7(this, action, ~)
% % ===========================
% % Reward v7: Hybrid v5 (signal) + v6 (low variance via Huber)
% % ===========================
% 
% persistent previousPosFlex inactiveSteps previousPhi stepCount
% persistent prevErrHuber
% 
% % Init (first time)
% if isempty(prevErrHuber), prevErrHuber = 0; end
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% if isempty(stepCount), stepCount = 0; end
% 
% % ---- RESET por episodio
% if this.c == 1
%     previousPosFlex = zeros(size(action));
%     inactiveSteps = 0;
%     previousPhi = 0;
%     stepCount = 0;
%     prevErrHuber = NaN;  % para no castigar primer step
% end
% 
% stepCount = stepCount + 1;
% 
% % ---------------------------
% % Parámetros (ajustables)
% % ---------------------------
% opts.gamma = 0.99;
% 
% % Señales principales
% kDist   = 1.0;     % penalización por distancia (suave)
% kProg   = 40;      % ganancia de progreso (v5)
% kShape  = 12;      % ganancia PBRS (más suave que v5 base=20)
% clipLimit = 20;    % clipping tanh
% 
% % Huber
% deltaHuber = 0.05; % umbral Huber (en espacio [0,1]) -> 0.03..0.08 típico
% 
% % Bonuses
% thrPrec = 0.03;
% thrStd  = 0.01;
% precisionBonus = 3;
% stabilityBonus = 2;
% successBonus   = 50;
% 
% % Reward por motor (v5 base)
% rewards = struct( ...
%     'dirInverse', -3, ...
%     'wrongStop', -6, ...
%     'goodMove', 15, ...
%     'goodMove2', 2, ...
%     'inactivityPenalty', -2, ...
%     'moveIncentive', 4, ...
%     'smoothnessPenalty', -2, ...
%     'efficiencyBonus', 3 ...
% );
% 
% rewardVector = zeros(1, length(action));
% 
% % ---------------------------
% % Estado actual (referencia y prótesis)
% % ---------------------------
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% 
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% % ---------------------------
% % Recompensas por motor (dirección + suavidad + eficiencia)
% % ---------------------------
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
% 
%     if abs(slope) > 0.01 && abs(slope) < 0.4
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% previousPosFlex = posFlex;
% 
% % ---------------------------
% % Inactividad (v5)
% % ---------------------------
% distance = abs(posFlex - flexConv(end, :));
% 
% if all(action == 0) && any(distance > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     penalty = rewards.inactivityPenalty * inactiveSteps;
%     rewardVector = rewardVector + penalty;
% else
%     inactiveSteps = 0;
% end
% 
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% % ---------------------------
% % Error robusto (Huber) (v6 style)
% % ---------------------------
% diff = (posFlex - flexConv(end, :));  % [-1,1]
% absDiff = abs(diff);
% 
% huberPerMotor = zeros(1,4);
% for i=1:4
%     if absDiff(i) <= deltaHuber
%         huberPerMotor(i) = 0.5 * (diff(i)^2);
%     else
%         huberPerMotor(i) = deltaHuber * (absDiff(i) - 0.5*deltaHuber);
%     end
% end
% errHuber = mean(huberPerMotor);  % escalar robusto
% 
% % Guardar métricas (para tus logs del env)
% this.meanDistStep = mean(distance);     % útil para comparar con lo anterior
% this.mseStep      = mean(diff.^2);
% this.successStep  = all(distance < thrPrec);
% 
% % ---------------------------
% % Penalización base por distancia (suave)
% % ---------------------------
% rewardVector = rewardVector - kDist * sqrt(distance);
% 
% % ---------------------------
% % Bonuses (v5)
% % ---------------------------
% precisionMask = distance < thrPrec;
% rewardVector(precisionMask) = rewardVector(precisionMask) + precisionBonus;
% 
% if std(distance) < thrStd
%     rewardVector = rewardVector + stabilityBonus;
% end
% 
% baseReward = sum(rewardVector);
% 
% if all(distance < thrPrec)
%     baseReward = baseReward + successBonus;
% end
% 
% % ---------------------------
% % Progreso (v5) pero con errHuber (v6)
% % ---------------------------
% if isnan(prevErrHuber)
%     progressTerm = 0;
% else
%     progressTerm = kProg * (prevErrHuber - errHuber);
% end
% prevErrHuber = errHuber;
% 
% % Clamp del progreso para bajar varianza
% progressTerm = max(min(progressTerm, 5), -5);
% 
% % ---------------------------
% % PBRS (v5) con potencial robusto
% % ---------------------------
% phiCurrent = -log(1 + errHuber);
% 
% % Decaimiento suave de shaping (opcional)
% shapingGain = kShape * exp(-0.0002 * stepCount);
% 
% shapingTerm = shapingGain * (opts.gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% % ---------------------------
% % Reward final + clipping (v5)
% % ---------------------------
% rewardRaw = baseReward + progressTerm + shapingTerm;
% reward = clipLimit * tanh(rewardRaw / clipLimit);
% 
% % Debug cada 10 steps
% if mod(this.c,10)==0
%     fprintf("c=%d base=%.3f prog=%.3f shape=%.3f raw=%.3f final=%.3f | errHuber=%.4f | meanDist=%.4f\n", ...
%         this.c, baseReward, progressTerm, shapingTerm, rewardRaw, reward, errHuber, mean(distance));
% end
% 
% end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%VERSION 8 (PBRS con Huber loss + clipping mas grande + se baja el base reward y se aumenta el progressTerm)  %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [reward, rewardVector, action] = legacy_distanceRewarding_v8(this, action, ~)
% ===========================
% Reward v8: Tracking-dominant (Progress + PBRS) + Low-variance (Huber)
% ===========================

% persistent previousPosFlex inactiveSteps previousPhi stepCount
% persistent prevErrHuber
% 
% % Init (first time)
% if isempty(prevErrHuber), prevErrHuber = 0; end
% if isempty(previousPosFlex), previousPosFlex = zeros(size(action)); end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% if isempty(previousPhi), previousPhi = 0; end
% if isempty(stepCount), stepCount = 0; end
% 
% % ---- RESET por episodio
% if this.c == 1
%     previousPosFlex = zeros(size(action));
%     inactiveSteps = 0;
%     previousPhi = 0;
%     stepCount = 0;
%     prevErrHuber = NaN;  % no castigar primer step
% end
% 
% stepCount = stepCount + 1;
% 
% % ---------------------------
% % Parámetros (v8)
% % ---------------------------
% gamma = 0.99;
% 
% % Pesos globales (CLAVE)
% wBase = 0.25;      % << reduce dominio del baseReward (0.2..0.35)
% kProg = 120;       % << progreso manda (80..160)
% kShape = 30;       % << PBRS más fuerte (20..40)
% clipLimit = 25;    % clipping un poco más alto
% 
% % Huber (robusto)
% deltaHuber = 0.05;
% 
% % Penalización distancia base
% kDist = 0.7;       % reduce presión por distancia directa (0.5..1.0)
% 
% % Bonus/umbrales
% thrPrec = 0.03;
% thrStd  = 0.01;
% precisionBonus = 2;     % baja un poco para no dominar
% stabilityBonus = 1.5;
% successBonus   = 40;
% 
% % Anti-cheating cuando empeora
% badProgressPenalty = 3;  % penaliza si dErr<0 (2..6)
% 
% % Reward por motor (menos agresivo que v5)
% rewards = struct( ...
%     'dirInverse', -2, ...
%     'wrongStop', -4, ...
%     'goodMove', 8, ...
%     'goodMove2', 1, ...
%     'inactivityPenalty', -1.5, ...
%     'moveIncentive', 2, ...
%     'smoothnessPenalty', -1.5, ...
%     'efficiencyBonus', 2 ...
% );
% 
% rewardVector = zeros(1, length(action));
% 
% % ---------------------------
% % Estado actual (referencia y prótesis)
% % ---------------------------
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% 
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% distance = abs(posFlex - flexConv(end, :));
% diff = (posFlex - flexConv(end, :));
% 
% % ---------------------------
% % Recompensas por motor (dirección + suavidad + eficiencia)
% % ---------------------------
% for i = 1:length(action)
%     if posFlex(i) < flexConv(end, i)
%         correctAction = 1;
%     elseif posFlex(i) > flexConv(end, i)
%         correctAction = -1;
%     else
%         correctAction = 0;
%     end
% 
%     if action(i) == correctAction
%         rewardVector(i) = rewards.goodMove * (action(i) ~= 0) + rewards.goodMove2 * (action(i) == 0);
%     elseif action(i) == 0
%         rewardVector(i) = rewards.wrongStop;
%     else
%         rewardVector(i) = rewards.dirInverse;
%     end
% 
%     slope = posFlex(i) - previousPosFlex(i);
%     rewardVector(i) = rewardVector(i) + rewards.smoothnessPenalty * sqrt(abs(slope));
% 
%     if abs(slope) > 0.01 && abs(slope) < 0.4
%         rewardVector(i) = rewardVector(i) + rewards.efficiencyBonus;
%     end
% end
% previousPosFlex = posFlex;
% 
% % ---------------------------
% % Inactividad (suave)
% % ---------------------------
% if all(action == 0) && any(distance > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     rewardVector = rewardVector + rewards.inactivityPenalty * inactiveSteps;
% else
%     inactiveSteps = 0;
% end
% 
% if any(action ~= 0)
%     rewardVector = rewardVector + rewards.moveIncentive;
% end
% 
% % ---------------------------
% % Error robusto Huber (tracking core)
% % ---------------------------
% absDiff = abs(diff);
% huberPerMotor = zeros(1,4);
% for i=1:4
%     if absDiff(i) <= deltaHuber
%         huberPerMotor(i) = 0.5 * (diff(i)^2);
%     else
%         huberPerMotor(i) = deltaHuber * (absDiff(i) - 0.5*deltaHuber);
%     end
% end
% errHuber = mean(huberPerMotor);
% 
% % Guardar métricas del env (si ya tienes esas properties públicas)
% this.meanDistStep = mean(distance);
% this.mseStep      = mean(diff.^2);
% this.successStep  = all(distance < thrPrec);
% 
% % ---------------------------
% % Penalización por distancia (suave)
% % ---------------------------
% rewardVector = rewardVector - kDist * sqrt(distance);
% 
% % ---------------------------
% % Bonuses (pequeños)
% % ---------------------------
% precisionMask = distance < thrPrec;
% rewardVector(precisionMask) = rewardVector(precisionMask) + precisionBonus;
% 
% if std(distance) < thrStd
%     rewardVector = rewardVector + stabilityBonus;
% end
% 
% baseReward = sum(rewardVector);
% 
% if all(distance < thrPrec)
%     baseReward = baseReward + successBonus;
% end
% 
% % <<<<<< CLAVE: baseReward reducido
% baseReward = wBase * baseReward;
% 
% % ---------------------------
% % Progreso (manda)
% % ---------------------------
% if isnan(prevErrHuber)
%     dErr = 0;
%     progressTerm = 0;
% else
%     dErr = (prevErrHuber - errHuber);   % positivo=mejora
%     progressTerm = kProg * dErr;
% end
% prevErrHuber = errHuber;
% 
% % clamp más grande (para que sí pese)
% progressTerm = max(min(progressTerm, 10), -10);
% 
% % penaliza explícitamente cuando empeora
% if dErr < 0
%     progressTerm = progressTerm - badProgressPenalty;
% end
% 
% % ---------------------------
% % PBRS (fuerte)
% % ---------------------------
% phiCurrent = -log(1 + errHuber);
% 
% % decaimiento suave para no dominar al final
% shapingGain = kShape * exp(-0.00015 * stepCount);
% 
% shapingTerm = shapingGain * (gamma * phiCurrent - previousPhi);
% previousPhi = phiCurrent;
% 
% % ---------------------------
% % Reward final + clipping
% % ---------------------------
% rewardRaw = baseReward + progressTerm + shapingTerm;
% reward = clipLimit * tanh(rewardRaw / clipLimit);
% 
% % Debug cada 10 steps
% if mod(this.c,10)==0
%     fprintf("c=%d base=%.2f prog=%.2f shape=%.2f raw=%.2f final=%.2f | errH=%.4f dErr=%.4f meanDist=%.4f\n", ...
%         this.c, baseReward, progressTerm, shapingTerm, rewardRaw, reward, errHuber, dErr, mean(distance));
% end
% 
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%% VERSION 9 (diatancia absoluta y dErr)  %%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %    function [reward, rewardVector, action] = reward_v9_progressOnly(this, action, ~)
% % ==========================================================
% % Reward V9 (firma compatible):
% % Reward basado en PROGRESO: dErr = err(t-1) - err(t)
% %
% % IMPORTANTE:
% % - NO escribe this.meanDistStep / this.mseStep / this.successStep
% %   (eso debe hacerse en step.m).
% % - Esta función es ROBUSTA si flexConverted/adjustEnc aún están vacíos.
% % ==========================================================
% 
% persistent prevErr inactiveSteps
% 
% if isempty(prevErr), prevErr = NaN; end
% if isempty(inactiveSteps), inactiveSteps = 0; end
% 
% % Reset por episodio
% if this.c == 1
%     prevErr = NaN;
%     inactiveSteps = 0;
% end
% 
% % -----------------------------
% % Obtener q_ref y q de forma segura
% % -----------------------------
% % q_ref (referencia)
% if isempty(this.flexConverted) || size(this.flexConverted,1) < 1
%     q_ref = zeros(1,4);
% else
%     q_ref = this.flexConverted(end, :);
% end
% 
% % q (estado actual)
% if isempty(this.adjustEnc) || size(this.adjustEnc,1) < 1
%     q = zeros(1,4);
% else
%     q = this.adjustEnc(end, :);
% end
% 
% % -----------------------------
% % Error y progreso
% % -----------------------------
% absErrVec = abs(q - q_ref);
% err = mean(absErrVec);
% 
% % dErr: positivo es bueno (error disminuyó)
% if isnan(prevErr)
%     dErr = 0;                 % no castigar primer step
% else
%     dErr = prevErr - err;
% end
% prevErr = err;
% 
% % -----------------------------
% % Hiperparámetros (ajustables)
% % -----------------------------
% kProgress    = 80;     % 40..120
% kInactivity  = 1.5;
% thrSuccess   = 0.03;
% successBonus = 10;
% clipLimit    = 10;
% 
% % -----------------------------
% % Penalización por inactividad
% % -----------------------------
% if all(action == 0) && any(absErrVec > 0.05)
%     inactiveSteps = inactiveSteps + 1;
%     inactivityPenalty = kInactivity * inactiveSteps;
% else
%     inactiveSteps = 0;
%     inactivityPenalty = 0;
% end
% 
% % -----------------------------
% % Bonus por éxito
% % -----------------------------
% successStep = all(absErrVec < thrSuccess);
% bonus = successBonus * double(successStep);
% 
% % -----------------------------
% % Reward final
% % -----------------------------
% rewardRaw = kProgress * dErr + bonus - inactivityPenalty;
% reward    = clipLimit * tanh(rewardRaw / clipLimit);
% 
% % rewardVector (compatible 1x4)
% rewardVector = ones(1, numel(action)) * reward;
% 
% end

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%% VERSION 10 %%%%%%%%%%%%%%%%%%
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %function [reward, rewardVector, action] = reward_v10_lyapunov_window(this, action, ~)
% % ==========================================================
% % Reward V10: Lyapunov progress + Moving Window smoothing
% % - Objetivo: alinear reward con Mean|q - qref| y MSE
% % - r_prog = k * (V_prev - V_curr), V = e^2
% % - e = mean(abs(q - qref))
% % ==========================================================
% 
% persistent prevV prevE stepCount
% persistent inactiveSteps prevPosFlex
% persistent errWindow
% 
% % -------------------------
% % Reset por episodio
% % -------------------------
% if isempty(stepCount); stepCount = 0; end
% if this.c == 1
%     stepCount = 0;
%     prevV = NaN;
%     prevE = NaN;
% 
%     inactiveSteps = 0;
%     prevPosFlex = zeros(size(action));
% 
%     % ventana de error (moving window)
%     W = 10;  % ventana (5..20 recomendado)
%     errWindow = NaN(1, W);
% end
% 
% stepCount = stepCount + 1;
% 
% % -------------------------
% % Parámetros principales
% % -------------------------
% opts.gamma = 0.99;
% 
% % Pesos (ajustables)
% kProgress   = 200;   % escala del progreso Lyapunov (50..400)
% kErrorAbs   = 5;     % penalización base por error medio (1..10)
% kSmooth     = 2;     % penalización por cambio brusco (0.5..5)
% kInactive   = 1.0;   % penalización por inactividad acumulada
% kMoveBonus  = 0.5;   % pequeño bonus por moverse
% successBonus = 20;   % bonus si llega cerca (10..80)
% 
% % Umbrales
% epsNear = 0.03;      % “cerca” por motor
% epsFar  = 0.06;      % “lejos” para inactividad
% 
% % Clipping
% clipLimit = 20;
% 
% rewardVector = zeros(1, numel(action));
% 
% % -------------------------
% % Obtener referencia y estado actual
% % -------------------------
% if this.c == 1
%     flexConv = this.flexJoined_scaler(reduceFlexDimension(this.flexData));
% else
%     flexConv = this.flexConvertedLog{this.c - 1};
% end
% 
% pos = this.motorData(end, :);
% posFlex = this.flexJoined_scaler(encoder2Flex(pos));
% 
% % Error por motor y global
% err = abs(posFlex - flexConv(end, :));      % vector error motor
% e = mean(err);                               % error global mean(|q-qref|)
% mse = mean((posFlex - flexConv(end, :)).^2); % MSE global
% 
% % Guardar métricas para logs (esto lo lee step.m)
% this.meanDistStep = e;
% this.mseStep = mse;
% 
% % Success step: "éxito" si todos los motores están cerca
% this.successStep = all(err < epsNear);
% 
% % -------------------------
% % Ventana deslizante (suavizado del error)
% % -------------------------
% errWindow = [errWindow(2:end), e];
% eWin = mean(errWindow, 'omitnan');  % error suavizado por ventana
% V = eWin^2;                         % energía Lyapunov basada en ventana
% 
% % -------------------------
% % 1) Reward de progreso Lyapunov
% % -------------------------
% if isnan(prevV)
%     r_prog = 0;  % no castigar primer step
% else
%     r_prog = kProgress * (prevV - V);
% end
% prevV = V;
% 
% % -------------------------
% % 2) Penalización por error absoluto
% % -------------------------
% r_err = -kErrorAbs * eWin;
% 
% % -------------------------
% % 3) Penalización por suavidad / cambios bruscos
% % -------------------------
% delta = posFlex - prevPosFlex;
% r_smooth = -kSmooth * mean(abs(delta));
% prevPosFlex = posFlex;
% 
% % -------------------------
% % 4) Penalización por inactividad cuando está lejos
% % -------------------------
% if all(action == 0) && any(err > epsFar)
%     inactiveSteps = inactiveSteps + 1;
% else
%     inactiveSteps = 0;
% end
% r_inactive = -kInactive * inactiveSteps;
% 
% % -------------------------
% % 5) Bonus por moverse (muy pequeño)
% % -------------------------
% r_move = kMoveBonus * any(action ~= 0);
% 
% % -------------------------
% % 6) Bonus por éxito
% % -------------------------
% r_success = 0;
% if all(err < epsNear)
%     r_success = successBonus;
% end
% 
% % -------------------------
% % Reward final
% % -------------------------
% rewardRaw = r_prog + r_err + r_smooth + r_inactive + r_move + r_success;
% 
% % Clipping suave
% reward = clipLimit * tanh(rewardRaw / clipLimit);
% 
% % rewardVector por motor (opcional para debug)
% rewardVector(:) = reward / numel(action);
% 
% % Debug opcional
% % if mod(this.c, 10)==0
% %     fprintf("c=%d eWin=%.4f V=%.4f r_prog=%.3f raw=%.3f final=%.3f\n", ...
% %         this.c, eWin, V, r_prog, rewardRaw, reward);
% % end
% 
% end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%% VERSION 12  (Wen 2019, Borkowska 2022, etc.)%%%%%%%%%%%%%
% % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %function [reward, rewardVector, actionOut] = progress_shaping_reward_v1(this, action, ~)
% 
% persistent prevMeanDist previousAction successCounter
% 
% % Acción de salida por defecto
% actionOut = action;
% 
% % Inicialización
% if isempty(prevMeanDist),  prevMeanDist = NaN; end
% if isempty(previousAction), previousAction = zeros(size(action)); end
% if isempty(successCounter), successCounter = 0; end
% 
% % Reset por episodio
% if this.c <= 1
%     prevMeanDist = NaN;
%     previousAction = zeros(size(action));
%     successCounter = 0;
% end
% 
% %% ===== 1) Variables reales del entorno =====
% q     = this.adjustEnc(end,:);       % encoder normalizado
% q_ref = this.flexConverted(end,:);   % referencia glove normalizada
% 
% e = q - q_ref;
% e_t = mean(abs(e));                  % error medio absoluto actual
% 
% if isnan(prevMeanDist)
%     delta_e = 0;
% else
%     delta_e = prevMeanDist - e_t;    % >0 mejora
% end
% 
% %% ===== 2) Hiperparámetros =====
% Kp = 40;     % progreso MUCHO más fuerte
% Kg = 0.3;    % meta más débil
% Ke = 0.02;   % menos castigo por esfuerzo
% Kj = 0.02;   % menos castigo por jerk
% Kd = 2.0;
% 
% successThreshold = 0.10;   % más relajado que el de métricas
% holdSteps = 3;             % pasos consecutivos en éxito
% successBonus = 30;
% holdBonus = 15;
% 
% %% ===== 3) Términos de reward =====
% 
% % (a) progreso: bajar el error
% r_progress = Kp * delta_e;
% 
% % (b) cercanía a la meta
% r_goal = -Kg * e_t;
% 
% % (c) castigo si empeora claramente
% if delta_e < -0.01
%     r_diverge = Kd * abs(delta_e);
% else
%     r_diverge = 0;
% end
% 
% % (d) castigo por esfuerzo de control
% r_effort = Ke * mean(abs(action));
% 
% % (e) castigo por cambios bruscos de acción
% r_jerk = Kj * mean(abs(action - previousAction));
% 
% % (f) bono por éxito
% if all(abs(e) < successThreshold)
%     successCounter = successCounter + 1;
%     r_success = successBonus;
% else
%     successCounter = 0;
%     r_success = 0;
% end
% 
% if successCounter >= holdSteps
%     r_success = r_success + holdBonus;
% end
% 
% %% ===== 4) Reward final =====
% reward = r_progress + r_goal + r_success - r_diverge - r_effort - r_jerk;
% 
% % clipping para estabilizar DQN
% reward = max(min(reward, 25), -25);
% 
% %% ===== 5) Reward vector (solo informativo) =====
% rewardVector = [ ...
%     r_progress, ...
%     r_goal, ...
%     r_success, ...
%    -r_diverge, ...
%    -r_effort, ...
%    -r_jerk];
% 
% %% ===== 5.1) Debug de componentes =====
% if mod(this.c,5) == 0 || this.c == 1
%     fprintf(['[RW DBG] step=%d | progress=% .4f | goal=% .4f | success=% .4f | ' ...
%              'divergence=% .4f | effort=% .4f | jerk=% .4f | total=% .4f\n'], ...
%              this.c, r_progress, r_goal, r_success, -r_diverge, -r_effort, -r_jerk, reward);
% end
% %% ===== 6) Actualizar memoria =====
% prevMeanDist = e_t;
% previousAction = action;
% 
% end



% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % %%%%%%%%%%%%%%%% VERSION 13  (Wen 2019, Borkowska 2022, etc.) optimizado %%%%%%%%%%%%%%
% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [reward, rewardVector, actionOut] = legacy_distanceRewarding(this, action, ~)
% 
% persistent prevMeanDist previousAction successCounter
% 
% actionOut = action;
% 
% if isempty(prevMeanDist), prevMeanDist = NaN; end
% if isempty(previousAction), previousAction = zeros(size(action)); end
% if isempty(successCounter), successCounter = 0; end
% 
% if this.c <= 1
%     prevMeanDist = NaN;
%     previousAction = zeros(size(action));
%     successCounter = 0;
% end
% 
% %% Variables reales del entorno
% q     = this.adjustEnc(end,:);
% q_ref = this.flexConverted(end,:);
% 
% e = q - q_ref;
% e_t = mean(abs(e));
% 
% if isnan(prevMeanDist)
%     delta_e = 0;
% else
%     delta_e = prevMeanDist - e_t;
% end
% 
% %% Hiperparámetros reescalados
% Kp = 40;      % progreso fuerte
% Kg = 0.25;    % castigo por distancia mucho menor
% Ke = 0.02;    % menor castigo por esfuerzo
% Kj = 0.02;    % menor castigo por jerk
% Kd = 0.0;     % divergencia desactivada temporalmente
% 
% successThreshold = 0.10;
% holdSteps = 3;
% successBonus = 30;
% holdBonus = 15;
% 
% %% Términos
% r_progress = Kp * delta_e;
% r_goal     = -Kg * e_t;
% 
% r_diverge = 0;
% r_effort  = Ke * mean(abs(action));
% r_jerk    = Kj * mean(abs(action - previousAction));
% 
% if all(abs(e) < successThreshold)
%     successCounter = successCounter + 1;
%     r_success = successBonus;
% else
%     successCounter = 0;
%     r_success = 0;
% end
% 
% if successCounter >= holdSteps
%     r_success = r_success + holdBonus;
% end
% 
% %% Reward total
% reward = r_progress + r_goal + r_success - r_diverge - r_effort - r_jerk;
% reward = max(min(reward, 25), -25);
% 
% %% Vector informativo
% rewardVector = [ ...
%     r_progress, ...
%     r_goal, ...
%     r_success, ...
%    -r_diverge, ...
%    -r_effort, ...
%    -r_jerk];
% 
% %% Debug
% if mod(this.c,5) == 0 || this.c == 1
%     fprintf(['[RW DBG V2] step=%d | progress=% .4f | goal=% .4f | success=% .4f | ' ...
%              'divergence=% .4f | effort=% .4f | jerk=% .4f | total=% .4f\n'], ...
%              this.c, r_progress, r_goal, r_success, -r_diverge, -r_effort, -r_jerk, reward);
% end
% 
% prevMeanDist = e_t;
% previousAction = action;
% 
% end



% % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % %%%%%%%%%%%%%%%% VERSION 14 REWARD SIMPLE ERR_PREV - ERR_ACT %%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %function [reward, rewardVector, action] = legacy_distanceRewarding(this, action)
% 
%     % ============================================================
%     % SIMPLE PROGRESS REWARD (diagnostic version)
%     % reward = previous error norm - current error norm
%     % Positive reward if error decreases
%     % ============================================================
% 
%     persistent prevErrNorm
% 
%     % -------- reset por episodio --------
%     if this.c == 1
%         prevErrNorm = NaN;
%     end
% 
%     % -------- estado actual --------
%     q     = this.adjustEnc(end,:);
%     q_ref = this.flexConverted(end,:);
% 
%     err = q - q_ref;
% 
%     currErrNorm = norm(err);
% 
%     % -------- reward --------
%     if isnan(prevErrNorm)
%         reward = 0;
%     else
%         reward = prevErrNorm - currErrNorm;
%     end
% 
%     % guardar error para siguiente step
%     prevErrNorm = currErrNorm;
% 
%     % -------- rewardVector compatible con pipeline --------
%     rewardVector = reward * ones(1,length(action));
% 
%     % -------- debug opcional --------
%     if mod(this.c,5) == 1 || this.c == 1
%         fprintf('[RW SIMPLE] step=%d | errNorm=%.4f | reward=%.4f\n', ...
%             this.c, currErrNorm, reward);
%     end
% 
%     end
% 
% % % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % % % %%%%%%%%%%%%%%%% VERSION 15 REWARD SIMPLE ERR_PREV - ERR_ACT + BONUS %%%%%%%%%%%%%%%%
% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function [reward, rewardVector, action] = legacy_distanceRewarding(this, action)
% ============================================================
% Reward simple + terminal bonus
% ------------------------------------------------------------
% 1) Reward principal: progreso en reducción del error
%       reward = prevErrNorm - currErrNorm
%
% 2) Bonus por "near success"
% 3) Bonus mayor por "strict success"
%
% Compatible con pipeline:
%   function [reward, rewardVector, action] = legacy_distanceRewarding(this, action)
% ============================================================

    persistent prevErrNorm prevNearFlag prevSuccessFlag

    % --------------------------------------------------------
    % Reset por episodio
    % --------------------------------------------------------
    if this.c == 1 || isempty(prevErrNorm)
        prevErrNorm    = NaN;
        prevNearFlag   = false;
        prevSuccessFlag = false;
    end

    % --------------------------------------------------------
    % Estado actual
    % --------------------------------------------------------
    q     = this.adjustEnc(end,:);      % estado actual de prótesis
    q_ref = this.flexConverted(end,:);  % referencia
    err   = q - q_ref;
    absErr = abs(err);

    currErrNorm = norm(err);

    % --------------------------------------------------------
    % Umbrales
    % --------------------------------------------------------
    thrSuccess = 0.20;   % mismo criterio estricto que estás usando en step
    thrNear    = 0.30;   % zona intermedia más permisiva

    isNear    = all(absErr < thrNear);
    isSuccess = all(absErr < thrSuccess);

    % --------------------------------------------------------
    % 1) Reward por progreso
    % --------------------------------------------------------
    if isnan(prevErrNorm)
        progressReward = 0;
    else
        progressReward = prevErrNorm - currErrNorm;
    end

    % --------------------------------------------------------
    % 2) Bonus terminal / por región objetivo
    % --------------------------------------------------------
    nearBonus = 0;
    successBonus = 0;

    % Bonus solo cuando entra por primera vez a la región
    if isNear && ~prevNearFlag
        nearBonus = 0.5;
    end

    if isSuccess && ~prevSuccessFlag
        successBonus = 2.0;
    end

    % Bonus de mantenimiento suave si ya está dentro
    if isSuccess
        successBonus = successBonus + 0.2;
    elseif isNear
        nearBonus = nearBonus + 0.05;
    end

    % --------------------------------------------------------
    % Reward total
    % --------------------------------------------------------
    reward = progressReward + nearBonus + successBonus;

    % --------------------------------------------------------
    % Vector de reward compatible con tu pipeline
    % --------------------------------------------------------
    rewardVector = reward * ones(1, length(action));

    % --------------------------------------------------------
    % Actualizar memoria para siguiente step
    % --------------------------------------------------------
    prevErrNorm     = currErrNorm;
    prevNearFlag    = isNear;
    prevSuccessFlag = isSuccess;

    % --------------------------------------------------------
    % Debug opcional
    % --------------------------------------------------------
    if this.verbose && (this.c == 1 || mod(this.c,5) == 1)
        fprintf(['[RW SIMPLE+BONUS] step=%d | errNorm=%.4f | progress=%.4f | ' ...
                 'near=%d | success=%d | nearBonus=%.2f | successBonus=%.2f | total=%.4f\n'], ...
                 this.c, currErrNorm, progressReward, ...
                 isNear, isSuccess, nearBonus, successBonus, reward);
    end
end
% % % % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
