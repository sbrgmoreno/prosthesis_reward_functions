function [trajectory, phaseIdxOut, phaseDirOut, phaseSpOut] = prosthesis_simulator( ...
    initial_position, speeds, duration, sampling_period, phaseIdxIn, phaseDirIn, phaseSpIn)

%% Input Validation
arguments
    initial_position (1, 4) double {mustBeReal}
    speeds           (1, 4) double {mustBeInRange(speeds, -255, 255)}
    duration         (1, 1) double {mustBePositive}
    sampling_period  (1, 1) double {mustBePositive} = 0.1

    phaseIdxIn (1,4) double = NaN(1,4)
    phaseDirIn (1,4) string = strings(1,4)
    phaseSpIn  (1,4) double = zeros(1,4)
end

% Outputs init
phaseIdxOut = phaseIdxIn;
phaseDirOut = phaseDirIn;
phaseSpOut  = phaseSpIn;

% One or more internal points per env step
n_points = max(1, ceil(duration / sampling_period));

% Fixed phase step in curve-index units
phaseStep = 1;

trajectory = nan(n_points, 4);

for m = 1:4
    [trajectory(:,m), phaseIdxOut(m), phaseDirOut(m), phaseSpOut(m)] = predict_1dim( ...
        initial_position(m), speeds(m), m, n_points, phaseStep, ...
        phaseIdxOut(m), phaseDirOut(m), phaseSpOut(m));
end

end


function [t_i, phaseOut, dirOut, spIdxOut] = predict_1dim( ...
    pos, speed, motorIdx, n_points, phaseStep, phaseIn, dirPrev, spPrev)

persistent sim_params tail_length PATTERN_CURVE
if isempty(sim_params) || isempty(PATTERN_CURVE)
    f = load("src/@SimController/fit_C2.mat");
    sim_params = f.params;
    tail_length = f.tail_length; %#ok<NASGU>

    PATTERN_CURVE = load("src/@SimController/pattern_curve.mat", "avgs").avgs;
end

SIM_SPEEDS = [0 64 96 128 160 192 224 256] - 1;
speeds_txt = ["sp_00","sp_3F","sp_5F","sp_7F","sp_9F","sp_BF","sp_DF","sp_FF"];

% Direction from speed
dir = "steady";
if speed > 0
    dir = "closing";
elseif speed < 0
    dir = "opening";
end
dirOut = dir;

% Stationary
if speed == 0
    t_i = repmat(pos, n_points, 1);
    phaseOut = phaseIn;
    spIdxOut = spPrev;
    return;
end

% Closest speed bucket
[~, spIdx] = min(abs(SIM_SPEEDS - abs(speed)));
spIdxOut = spIdx;

sp_txt = speeds_txt(spIdx);
motor_txt = sprintf("m_%d", motorIdx);

% Validate data availability
if ~isfield(sim_params, sp_txt) || ~isfield(sim_params.(sp_txt).(dir), motor_txt) || ...
   ~isfield(PATTERN_CURVE.(sp_txt).(dir), motor_txt)
    error('Missing waveform data for speed "%s" dir "%s" motor %d.', sp_txt, dir, motorIdx);
end

curve = PATTERN_CURVE.(sp_txt).(dir).(motor_txt).avg;
if isempty(curve)
    error('Empty curve for speed "%s" dir "%s" motor %d.', sp_txt, dir, motorIdx);
end

% Saturate pos to allowed limits
p = sim_params.(sp_txt).(dir).(motor_txt);
y_sat = max(min(pos, p.max_lim), p.min_lim);

% Init or continue phase
needInit = isnan(phaseIn) || (dirPrev ~= dir) || (spPrev ~= spIdxOut) || ...
           (phaseIn < 1) || (phaseIn > length(curve));

if needInit
    % Start at the closest point of the curve to current position
    [~, start_idx] = min(abs(curve - y_sat));
    phase = start_idx;
else
    phase = phaseIn;
end

% Direction of phase movement
if speed > 0
    signedStep = +phaseStep;
else
    signedStep = -phaseStep;
end

% Generate trajectory
t_i = zeros(n_points, 1);
for t = 1:n_points
    idxCurve = round(phase);
    idxCurve = min(max(idxCurve, 1), length(curve));
    t_i(t) = curve(idxCurve);

    phase = phase + signedStep;
end

phaseOut = phase;

end


% function [trajectory, phaseIdxOut, phaseDirOut, phaseSpOut] = prosthesis_simulator( ...
%     initial_position, speeds, duration, sampling_period, phaseIdxIn, phaseDirIn, phaseSpIn)
% %function trajectory = prosthesis_simulator( ...
% %    initial_position, speeds, duration, sampling_period)
% %prosthesis_simulator() simulates the prosthesis dynamics. It returns the
% %trajectory of the motors given an initial position, the speed and the
% %duration of the movement.
% %
% % # INPUTS
% %  initial_position     1-by-4 vector with the initial position of the
% %                       motors
% %  speeds               1-by-4 vector with the speed of the motors
% %  duration             duration of the movement [seconds]
% %  sampling_period      sampling period of the simulation [seconds]
% %
% % # OUTPUTS
% %  trajectory           n-by-4 matrix with the trajectory of the motors
% %
% % # EXAMPLES
% %>> prosthesis_simulator([0 0 0 0], [0 0 0 0], 5)
% %
% %
% %>> prosthesis_simulator([0 0 0 0], [100 100 100 100], 1)
% %
% %
% %>> prosthesis_simulator([10 20 30 40], [-100 -255 100 -255], 1)
% 
% 
% %{
% Laboratorio de Inteligencia y Visin Artificial
% ESCUELA POLITCNICA NACIONAL
% Quito - Ecuador
% 
% autor: Jonathan Zea
% jonathan.a.zea@ieee.org
% 
% "I find that I don't understand things unless I try to program them."
% -Donald E. Knuth
% 
% 20 December 2023
% %}
% 
% %% Input Validation
% arguments
%     initial_position (1, 4) double {mustBeReal}
%     speeds (1, 4) double {mustBeInRange(speeds, -255, 255)}
%     duration (1, 1) double {mustBePositive}
%     sampling_period (1, 1) double {mustBePositive} = 0.1;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%7
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     phaseIdxOut = phaseIdxIn;
%     phaseDirOut = phaseDirIn;
%     phaseSpOut  = phaseSpIn;
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% end
% 
% % ------ auxs vars
% % n_points = round(duration/sampling_period); %****|||| ANTES DESCOMENTADAS
% % delta_ms = duration*1000/n_points;%****|||| ANTES DESCOMENTADAS
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% n_points = max(1, ceil(duration/sampling_period));
% delta_ms = max(1, round((sampling_period*1000)/10));  % avanza más lento (ajustable)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% % ------ loop
% trajectory = nan(n_points, 4);
% 
% %------------------ ANTES DESCOMENTADO ---------------------------
% % for i = 1:4
% %     trajectory(:, i) = predict_1dim(initial_position(i), speeds(i), ...
% %         i, n_points, delta_ms);
% % end
% %-------------------------------------------------------------------
% %%%%%%%%%%%%%%%%%    ^    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%    |    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%    |    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for m = 1:4
%     [trajectory(:,m), phaseIdxOut(m), phaseDirOut(m), phaseSpOut(m)] = predict_1dim( ...
%         initial_position(m), speeds(m), m, n_points, delta_ms, ...
%         phaseIdxOut(m), phaseDirOut(m), phaseSpOut(m));
% end
% %%%%%%%%%%%%%%%%% 4 LINEAS COMENTADAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% end
% 
% %% ########################################################################
% % function t_i = predict_1dim(pos, speed, i, n_points, delta_ms)
% % % returns the column vector with trayectory of a degree of freedom
% % 
% % persistent sim_params tail_length PATTERN_CURVE
% % if isempty(sim_params)
% %     f_name = "src/@SimController/fit_C2.mat";
% %     f = load(f_name);
% %     sim_params = f.params;
% %     tail_length = f.tail_length;
% % 
% %     f_name = "src/@SimController/pattern_curve.mat";
% %     PATTERN_CURVE = load(f_name, "avgs").avgs;
% % end
% % 
% % % ------ defaults
% % % from 0-31 does not move. From 32 64, first spped, 65 to 96 2nd, etc.
% % SIM_SPEEDS = [0 64 96 128 160 192 224 256] - 1;
% % speeds_txt = ["" "sp_3F" "sp_5F" "sp_7F" "sp_9F" "sp_BF" "sp_DF" "sp_FF"];
% % 
% % %--
% % if sign(speed) > 0
% %     dir = "closing";
% % elseif sign(speed) < 0
% %     dir = "opening";
% % else
% %     t_i = repmat(pos, n_points, 1); % at zero remains in the same position
% %     return
% % end
% % sp = abs(speed);
% % 
% % %-- get closest speed
% % for i_2 = 2:numel(SIM_SPEEDS)
% %     % [a sp b]
% %     b = SIM_SPEEDS(i_2);
% %     a = SIM_SPEEDS(i_2 - 1);
% % 
% %     if sp <= b
% %         if sp >= a
% %             % in the middle
% %             r = (sp - a)/(b - a);
% %             if r >= 0.5
% %                 sp = b;
% %             else
% %                 sp = a;
% %             end
% % 
% %         else
% %             % slower than the slowest!
% %             sp = 0;
% %         end
% %         break;
% %     end
% % end
% % 
% % %- slow speed
% % if sp == 0
% %     t_i = repmat(pos, n_points, 1); % at zero remains in the same position
% %     return;
% % end
% % 
% % 
% % %--- getting signal
% % sp_txt = speeds_txt(SIM_SPEEDS == sp);
% % m_txt = sprintf("m_%d", i);
% % 
% % % extracting
% % ws = sim_params.(sp_txt).(dir).(m_txt).ws;
% % min_l = sim_params.(sp_txt).(dir).(m_txt).min_lim;
% % max_l = sim_params.(sp_txt).(dir).(m_txt).max_lim;
% % 
% % y_sat = sat(pos, min_l, max_l);
% % 
% % % find init time in curve...
% % curve = PATTERN_CURVE.(sp_txt).(dir).(m_txt).avg;
% % 
% % for t = 1:numel(curve)
% %     if dir == "closing"
% %         if curve(t) >= y_sat
% %             break;
% %         end
% %     else
% %         if y_sat >= curve(t)
% %             break;
% %         end
% %     end
% % 
% % end
% % 
% % x_0 = tail_length + t;
% % 
% % %---
% % t_i = nan(n_points, 1); % prealloc
% % for t = 1:n_points
% %     t_i(t) = ws(x_0 + delta_ms*t);
% % end
% % 
% % end
% 
% 
% 
% function [t_i, phaseOut, dirOut, spIdxOut] = predict_1dim(pos, speed, i, n_points, delta_ms, phaseIn, dirPrev, spPrev)
% %function t_i = predict_1dim(pos, speed, i, n_points, delta_ms)
%     persistent sim_params tail_length PATTERN_CURVE
%     if isempty(sim_params) || isempty(PATTERN_CURVE)
%         % Load the simulation parameters and pattern curves
%         f_name = "src/@SimController/fit_C2.mat";
%         f = load(f_name);
%         sim_params = f.params;
%         tail_length = f.tail_length;
% 
%         f_name = "src/@SimController/pattern_curve.mat";
%         PATTERN_CURVE = load(f_name, "avgs").avgs;
%     end
% 
%     % Define the speed thresholds and corresponding labels
%     SIM_SPEEDS = [0 64 96 128 160 192 224 256] - 1;
%     speeds_txt = ["sp_00", "sp_3F", "sp_5F", "sp_7F", "sp_9F", "sp_BF", "sp_DF", "sp_FF"];
% 
%     % Determine the direction of movement based on speed
%     dir = "steady";
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     dirOut = dir    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
%     if speed > 0
%         dir = "closing";
%     elseif speed < 0
%         dir = "opening";
%     end
% 
%     % Handle stationary case
%     if speed == 0
%         t_i = repmat(pos, n_points, 1);
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         phaseOut = phaseIn;
%         spIdxOut = spPrev;
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         return;
%     end
% 
%     % Find the closest predefined speed setting
%     [~, idx] = min(abs(SIM_SPEEDS - abs(speed)));
%     sp_txt = speeds_txt(idx);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     spIdxOut = idx;    
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     motor_txt = sprintf("m_%d", i);
% 
%     % Check for waveform data existence
%     if ~isfield(sim_params, sp_txt) || ~isfield(sim_params.(sp_txt).(dir), motor_txt) || ...
%        ~isfield(PATTERN_CURVE.(sp_txt).(dir), motor_txt)
%         error('Waveform data for speed "%s" and direction "%s" is missing for motor %d.', sp_txt, dir, i);
%     end
% 
%     % Retrieve the waveform curve and ensure it exists
%     curve = PATTERN_CURVE.(sp_txt).(dir).(motor_txt).avg;
%     if isempty(curve)
%         error('Curve data is missing for speed "%s" and direction "%s" for motor %d.', sp_txt, dir, i);
%     end
% 
%     % Initialize trajectory calculation
%     t_i = zeros(n_points, 1);  % Preallocate output array
%     y_sat = max(min(pos, sim_params.(sp_txt).(dir).(motor_txt).max_lim), sim_params.(sp_txt).(dir).(motor_txt).min_lim);
% 
%     %------------------- ANTES DESCOMENTADO ----------------------------------------------------------------
%     % % Calculate starting point in the curve based on the current position
%     % start_idx = find(dir == "closing" & curve >= y_sat | dir == "opening" & curve <= y_sat, 1, 'first');
%     % if isempty(start_idx)
%     %     start_idx = 1;  % Default to starting at the beginning if not found
%     % end
%     %-------------------------------------------------------------------------------------------------------
% 
%     %%%%%%%%%%%%%%%%%% ^   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%% |   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%% |   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %[~, start_idx] = min(abs(curve - y_sat));   % índice más cercano a la posición actual
% 
%     % ====== PHASE INIT / CONTINUE ======
%     needInit = isnan(phaseIn) || dirPrev ~= dir || spPrev ~= spIdxOut;
% 
%     if needInit
%         % ubica el índice más cercano a la posición actual
%         [~, start_idx] = min(abs(curve - y_sat));
%         phase = tail_length + start_idx;
%     else
%         phase = phaseIn;
%     end
%     %%%%%%%%%%%%%%%%%% 5 LINEAS COMENTADAS    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
%     %x_0 = tail_length + start_idx; %****||| ANTES DESCOMENTADO
% 
%     % % Generate the motor trajectory respecting the bounds of the waveform
%     % for t = 1:n_points
%     %     %idx = round(x_0 + delta_ms * t); %||||ANTES DESCOMENTADO
%     %     %%%%%%%%%%%%%%%%%%%%%  ^    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %     %%%%%%%%%%%%%%%%%%%%%  |    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %     %%%%%%%%%%%%%%%%%%%%%  |    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % 
%     %     idx = round(x_0 + delta_ms * (t-1));  %primer punto donde exactamente estoy
%     %     %%%%%%%%%%%%%%%%%%%%%% 1 LINEA COMENTADA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % 
%     % 
%     % 
%     %     if idx > length(curve)
%     %         idx = length(curve);
%     %     elseif idx < 1
%     %         idx = 1;
%     %     end
%     %     t_i(t) = curve(idx);
%     % end
% 
%     %%%%%%%%%%%%%%%%%% ^   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%% |   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%%%%%%%%%%%%% |   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%     t_i = zeros(n_points,1);
% 
%     for t = 1:n_points
%         idxCurve = round(phase);      % índice actual
%         idxCurve = min(max(idxCurve,1), length(curve));
%         t_i(t) = curve(idxCurve);
% 
%         % Avanza phase para la siguiente muestra
%         phase = phase + delta_ms;     % delta_ms ya es "paso" en índices de la curva
%     end
% 
%     phaseOut = phase;
%     %%%%%%%%% 16 LINEAS COMENTADAS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% end
% 
% 
