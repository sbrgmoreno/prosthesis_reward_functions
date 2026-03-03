function analyze_sensitivity(env)
% analyze_sensitivity(env)
% Usa logs del Env para diagnosticar controlabilidad y calidad de señal.

T = env.c;
if isempty(T) || T < 2
    error("No hay suficientes steps en logs. Ejecuta al menos 1 episodio de sim() antes.");
end

% Recorta a steps reales
dErr = env.dErrLog(1:T);
eff  = env.effectNormLog(1:T);
agree = mean(env.dirAgreeLog(1:T,:), 2);  % % de motores con dirección correcta
errN = env.errNormLog(1:T);

figure; plot(dErr); grid on
title("dErr = err(t-1)-err(t) (positivo=mejora)");
xlabel("Step"); ylabel("dErr");

figure; plot(eff); grid on
title("EffectNorm = ||dq|| (si ~0 => acción afecta estado)");
xlabel("Step"); ylabel("||dq||");

figure; plot(errN); grid on
title("||error|| = ||q-qref||");
xlabel("Step"); ylabel("||e||");

figure; plot(agree); grid on
title("Direction agreement (0..1): acción en dirección correcta");
xlabel("Step"); ylabel("agreement");

% Correlaciones rápidas
fprintf("\n==== DIAGNOSTICO ====\n");
fprintf("Promedio ||dq|| (efecto acción): %.6f\n", mean(eff,'omitnan'));
fprintf("Promedio dErr (mejora): %.6f\n", mean(dErr,'omitnan'));
fprintf("%% steps con mejora (dErr>0): %.2f %%\n", 100*mean(dErr>0,'omitnan'));
fprintf("Agreement medio: %.3f\n", mean(agree,'omitnan'));

% Relación efecto -> mejora
valid = ~isnan(eff) & ~isnan(dErr);
if nnz(valid) > 5
    r = corr(eff(valid), dErr(valid));
    fprintf("Corr(||dq||, dErr): %.3f\n", r);
end

% Señal de problema: mucha acción sin efecto
deadSteps = eff < 1e-6;
fprintf("%% dead-zone (||dq|| ~ 0): %.2f %%\n", 100*mean(deadSteps,'omitnan'));

end




















%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% SENSITIVITY FOR DATA  %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% function analyze_sensitivity(env)
% % Analiza sensibilidad acción->estado usando logs del env (1 episodio)
% 
% T = env.c;  % steps ejecutados realmente
% 
% if isempty(T) || T <= 0
%     error("No hay steps ejecutados (env.c = %d). Primero ejecuta reset(env) y al menos un step(env, action).", T);
% end
% 
% % Usa errNormLog para construir dErr sin depender de dErrLog
% err = env.errNormLog(1:T);
% dErr = [0; err(1:end-1) - err(2:end)];
% 
% % Si existe dErrLog y tiene datos, úsalo; si no, usa el calculado
% if isprop(env,'dErrLog') && numel(env.dErrLog) >= T && any(~isnan(env.dErrLog(1:T)))
%     dErr = env.dErrLog(1:T);
% end
% 
% q   = env.qLog(1:T,:);
% qref= env.qRefLog(1:T,:);
% dq  = env.dqLog(1:T,:);
% aap = env.aAppliedLog(1:T,:);
% aMg = vecnorm(aap,2,2);
% 
% dqMg = env.effectNormLog(1:T);
% 
% dirAcc = mean(env.dirAgreeLog(1:T,:), 'all', 'omitnan');
% fprintf("Steps=%d | Directional accuracy=%.3f\n", T, dirAcc);
% 
% figure; plot(dqMg); grid on;
% title('||dq|| per step (movement magnitude)');
% xlabel('Step'); ylabel('norm(dq)');
% 
% figure; scatter(aMg, dqMg, '.'); grid on;
% title('Action magnitude vs movement magnitude');
% xlabel('||a_applied||'); ylabel('||dq||');
% 
% figure; plot(err); grid on;
% title('||e|| per step (tracking error norm)');
% xlabel('Step'); ylabel('norm(q-q_{ref})');
% 
% figure; plot(dErr); grid on;
% title('dErr = err(t-1)-err(t) (positive=good)');
% xlabel('Step'); ylabel('dErr');
% 
% end

