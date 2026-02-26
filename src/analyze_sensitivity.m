function analyze_sensitivity(env)
% Analiza sensibilidad acción->estado usando logs del env (1 episodio)

T = env.c;  % steps ejecutados realmente

if isempty(T) || T <= 0
    error("No hay steps ejecutados (env.c = %d). Primero ejecuta reset(env) y al menos un step(env, action).", T);
end

% Usa errNormLog para construir dErr sin depender de dErrLog
err = env.errNormLog(1:T);
dErr = [0; err(1:end-1) - err(2:end)];

% Si existe dErrLog y tiene datos, úsalo; si no, usa el calculado
if isprop(env,'dErrLog') && numel(env.dErrLog) >= T && any(~isnan(env.dErrLog(1:T)))
    dErr = env.dErrLog(1:T);
end

q   = env.qLog(1:T,:);
qref= env.qRefLog(1:T,:);
dq  = env.dqLog(1:T,:);
aap = env.aAppliedLog(1:T,:);
aMg = vecnorm(aap,2,2);

dqMg = env.effectNormLog(1:T);

dirAcc = mean(env.dirAgreeLog(1:T,:), 'all', 'omitnan');
fprintf("Steps=%d | Directional accuracy=%.3f\n", T, dirAcc);

figure; plot(dqMg); grid on;
title('||dq|| per step (movement magnitude)');
xlabel('Step'); ylabel('norm(dq)');

figure; scatter(aMg, dqMg, '.'); grid on;
title('Action magnitude vs movement magnitude');
xlabel('||a_applied||'); ylabel('||dq||');

figure; plot(err); grid on;
title('||e|| per step (tracking error norm)');
xlabel('Step'); ylabel('norm(q-q_{ref})');

figure; plot(dErr); grid on;
title('dErr = err(t-1)-err(t) (positive=good)');
xlabel('Step'); ylabel('dErr');

end

