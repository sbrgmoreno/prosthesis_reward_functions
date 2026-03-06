classdef SimController < handle
    % SimController mock class for Controller.
    % Includes a simulator of the prosthesis dynamics.
    % Requires the Timing object of the episode.

    %%
    properties (SetAccess=protected)
        vels   = [0 0 0 0];   % 1-by-4
        buffer = [0 0 0 0];

        tocStop;              % moment from which prosthesis stopped
        timing;               % object that mocks time

        sampling_period = 0.2; % seconds (will be overwritten by timing.period() in ctor)
        c0 = 0;               % counter of periods

        % ===== Phase memory (for curve-based simulator) =====
        phaseIdx = NaN(1,4);       % index/phase inside curve per motor
        lastDir = strings(1,4);   % "closing"/"opening"/"steady"
        lastSpIdx = zeros(1,4);     % speed bucket idx (1..8)
    end

    properties (Hidden=true)
        isConnected = false;
    end

    methods
        %% Constructor
        function obj = SimController(timing)
            obj.isConnected = true;

            obj.timing = timing;
            obj.c0 = timing.c;

            % Keep simulator sampling aligned with env period when possible
            try
                obj.sampling_period = timing.period();
            catch
                % keep default
            end

            % init phase state
            obj.phaseIdx(:) = NaN;
            obj.lastDir(:)   = "steady";
            obj.lastSpIdx(:) = 0;
        end

        %%
        function completed = closeHand(obj)
            completed = obj.sendAllSpeed(255, 255, 255, 255);
        end

        %%
        function completed = sendAllSpeed(obj, pwm1, pwm2, pwm3, pwm4)
            % IMPORTANT: do NOT call updatePos() here (step-based integration)
            obj.vels = [pwm1, pwm2, pwm3, pwm4];
            completed = true;
        end

        %%
        function completed = sendSpeed(obj, motor, pwm)
            % IMPORTANT: do NOT call updatePos() here (step-based integration)
            obj.vels(motor) = pwm;
            completed = true;
        end

        %%
        function completed = resetEncoder(obj, v1, v2, v3, v4)
            % Data Validation
            arguments
                obj
                v1(1,1) double {mustBeInteger} = 0
                v2(1,1) double {mustBeInteger} = 0
                v3(1,1) double {mustBeInteger} = 0
                v4(1,1) double {mustBeInteger} = 0
            end

            obj.updatePos();
            % obj.buffer(end + 1, :) = [v1 v2 v3 v4]; % *****||| ANTES COMENTADO
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % reset total (incluye fase) y fija nuevo estado
            obj.resetBuffer([v1 v2 v3 v4], true);
        
            % buffer arranca desde ese punto
            obj.buffer = [v1 v2 v3 v4];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            completed = true;
        end

        %%
        function completed = stop(obj)
            obj.updatePos();
            obj.vels = zeros(1, 4);
            completed = true;
        end

        %%
        function completed = stopMotor(obj, idxs)
            obj.updatePos();
            obj.vels(idxs) = 0;
            completed = true;
        end

        %%
        function completed = goHomePosition(obj, ~, ~, ~)
            obj.updatePos();
            %obj.resetBuffer(); %*****||| ANTES COMENTADO
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % reset total (incluye fase)
            obj.resetBuffer([0 0 0 0], true);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            obj.buffer = [0 0 0 0];
            completed = true;
        end

        %%
        function resetBuffer(obj, last_pos, resetPhase) % ANTES SIN resetPhase

            % %--------------------- ANTES DESCOMENTADO --------------------
            % % Data Validation
            % arguments
            %     obj
            %     last_pos (1, 4) double = [0 0 0 0]
            % end
            % 
            % obj.buffer = last_pos;
            % obj.c0 = obj.timing.c;
            % 
            % % reset phase memory (new episode / new read window)
            % obj.phaseIdx(:) = NaN;
            % obj.phaseDir(:) = "steady";
            % obj.phaseSp(:)  = 0;
            % %--------------------------------------------------------------
            %%%%%%%%%%%%%%%     ^      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%    |      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%    |      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            arguments
                obj
                last_pos   (1,4) double = [0 0 0 0];
                resetPhase (1,1) logical = true;
            end
        
            obj.buffer = last_pos;
            obj.c0 = obj.timing.c;
        
            if resetPhase
                obj.phaseIdx(:)  = NaN;
                obj.lastDir(:)   = "steady";
                obj.lastSpIdx(:) = 0;
        
                % Si también estás usando estos (recomendado)
                % if isprop(obj,'phaseDir'), obj.phaseDir(:) = "steady"; end
                % if isprop(obj,'phaseSp'),  obj.phaseSp(:)  = 0;       end
            end
            %%%%%%%%%%%% 9 LINEAS COMENTADAS      %%%%%%%%%%%%%%%%%%%%%%%%%
        end

        %%
        function data = read(obj)
            obj.updatePos();
            data = obj.buffer;
            %obj.resetBuffer(data(end, :)); %*****||| ANTES COMENTADO
            %%%%%%%%%%%%%  ^     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%  |     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%  |     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Recorta buffer al último punto, pero NO resetea la fase
            obj.resetBuffer(data(end,:), false);
            %%%%%%%%%%%1 LINEA COMENTADA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
    end

    methods (Access=protected)
        %%
        function updatePos(obj)
            % Calculates and updates trajectory until this moment on time
            % with the recorded speeds.

            cs = obj.timing.c - obj.c0;

            % If clock goes backwards (Timing reset), resync and skip integration
            if cs < 0
                warning("Timing went backwards (timing.c=%d, c0=%d). Resyncing.", obj.timing.c, obj.c0);
                obj.c0 = obj.timing.c;
                return;
            end

            % If counter didn't advance, integrate at least one period (RL step)
            if cs == 0
                cs = 1;
            end

            duration = cs * obj.timing.period();

            % Update c0 AFTER using the old value for integration
            obj.c0 = obj.timing.c;

            if isempty(duration)
                error("Duration can not be empty for simulator.")
            end

            % Integrate using curve-based simulator WITH phase memory
            [trajectory, obj.phaseIdx, obj.lastDir, obj.lastSpIdx] = SimController.prosthesis_simulator( ...
            obj.buffer(end,:), obj.vels, duration, obj.sampling_period, ...
            obj.phaseIdx, obj.lastDir, obj.lastSpIdx);

            obj.buffer = [obj.buffer; trajectory];

            % Debug print (optional)
            fprintf("timing.c=%d c0=%d cs=%d period=%.4f duration=%.4f\n", ...
                obj.timing.c, obj.c0, cs, obj.timing.period(), duration);
        end
    end

    methods (Static)
        [trajectory, phaseIdxOut, phaseDirOut, phaseSpOut] = prosthesis_simulator( ...
            initial_position, speeds, duration, sampling_period, phaseIdxIn, phaseDirIn, phaseSpIn);
    end
end







% classdef SimController < handle
%     %SimController mock class for Controller.
%     % Includes a simulator of the prosthesis dynamics.
%     % Requires the Timing object of the episode.
% 
%     %{
%     Laboratorio de Inteligencia y Visión Artificial
%     ESCUELA POLITÉCNICA NACIONAL
%     Quito - Ecuador
% 
%     autor: ztjona
%     jonathan.a.zea@ieee.org
% 
%     "I find that I don't understand things unless I try to program them."
%     -Donald E. Knuth
% 
%     %}
% 
%     %%
%     properties (SetAccess=protected)
%         vels = [0 0 0 0];%1-by-4
% 
%         buffer = [0 0 0 0];
% 
%         tocStop; % moment from which prosthesis stopped
% 
%         timing; % object that mocks time
% 
%         sampling_period = 0.2; % 0.14; % seconds
% 
%         c0 = 0; % counter of periods
% 
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%         phaseIdx = [NaN NaN NaN NaN];   % índice actual dentro de la curva por motor
%         lastDir  = strings(1,4);        % "closing"/"opening"/"steady"
%         lastSpIdx = zeros(1,4);         % idx de speed (para detectar cambios)
%         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
%     end
% 
%     properties (Hidden=true)
%         isConnected = false;
%     end
% 
%     methods
%         %% Constructor
%         % -----------------------------------------------------------------
%         function obj = SimController(timing)
%             %SimController(...)
%             %
% 
%             % # ----
%             obj.isConnected = true;
% 
%             obj.timing = timing;
%             obj.c0 = timing.c;
% 
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             obj.sampling_period = timing.period();  % iguala al periodo del env
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function completed = closeHand(obj)
% 
%             obj.sendAllSpeed(255, 255, 255, 255);
%             completed = true;
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function completed = sendAllSpeed(obj, pwm1, pwm2, pwm3, pwm4)
% 
%             %obj.updatePos();   %*****|||||||ANTES DESCOMENTADO ||||||*********
% 
%             obj.vels = [pwm1, pwm2, pwm3, pwm4];
%             completed = true;
%         end
%         %%
%         % -----------------------------------------------------------------
%         function completed = sendSpeed(obj, motor, pwm)
%             %obj.updatePos();      %*****|||||||ANTES DESCOMENTADO ||||||*********
% 
%             obj.vels(motor) = pwm;
%             completed = true;
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function completed = resetEncoder(obj, v1, v2, v3, v4)
%             % # ---- Data Validation
%             arguments
%                 obj
%                 v1(1,1) double {mustBeInteger} = 0;
%                 v2(1,1) double {mustBeInteger} = 0;
%                 v3(1,1) double {mustBeInteger} = 0;
%                 v4(1,1) double {mustBeInteger} = 0;
%             end
% 
%             obj.updatePos();
%             obj.buffer(end + 1, :) = [v1 v2 v3 v4];
%             completed = true;
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function completed = stop(obj)
%             obj.updatePos();
% 
%             obj.vels = zeros(1, 4);
%             completed = true;
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function completed = stopMotor(obj, idxs)
%             obj.updatePos();
% 
%             obj.vels(idxs) = 0;
%             completed = true;
%         end
%         %%
%         % -----------------------------------------------------------------
%         function completed = goHomePosition(obj, ~, ~, ~)
%             obj.updatePos();
% 
%             obj.resetBuffer();
% 
%             obj.buffer = [0 0 0 0];
%             completed = true;
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function resetBuffer(obj, last_pos)
%             % # ---- Data Validation
%             arguments
%                 obj
%                 last_pos   (1, 4) double = [0 0 0 0];
%             end
% 
%             obj.buffer = last_pos;
%             obj.c0 = obj.timing.c;
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%             obj.phaseIdx(:) = NaN;
%             obj.lastDir(:) = "steady";
%             obj.lastSpIdx(:) = 0;
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
%         end
% 
%         %%
%         % -----------------------------------------------------------------
%         function data = read(obj)
%             obj.updatePos();
% 
%             data = obj.buffer;
%             obj.resetBuffer(data(end, :));
%         end
%     end
% 
%     methods (Access=protected)
% 
%         %%
%         % -----------------------------------------------------------------
%         function updatePos(obj)
%             % Calculates and updates trajectory until this moment on time
%             % with the recorded speeds.
% 
%             % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % %%%%%%%%%%% NUEVO CORREGIDO               %%%%%%%%%%%%%%%%%%%%%
%             % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % cs = obj.timing.c - obj.c0;
%             % 
%             % % Si el contador no avanzó, integramos al menos 1 periodo (modo RL step)
%             % if cs == 0
%             %     cs = 1;
%             % end
%             % 
%             % duration = cs * obj.timing.period();
%             % obj.c0 = obj.timing.c;
%             % 
%             % if duration < 0
%             %     warning("Duration lower than 0.")
%             %     return;
%             % elseif isempty(duration)
%             %     error("Duration can not be empty for simulator.")
%             % end
% 
%             cs = obj.timing.c - obj.c0;
%             % Si el reloj retrocede (por reset del Timing), re-sincroniza y no integres
%             if cs < 0
%                 warning("Timing went backwards (timing.c=%d, c0=%d). Resyncing.", obj.timing.c, obj.c0);
%                 obj.c0 = obj.timing.c;
%                 return;
%             end
% 
%             % Si no avanzó, integramos al menos 1 periodo (modo RL step)
%             if cs == 0
%                 cs = 1;
%             end
% 
%             duration = cs * obj.timing.period();
% 
%             % Integra usando el c0 anterior; luego actualiza c0 al valor actual
%             obj.c0 = obj.timing.c;
% 
%             if isempty(duration)
%                 error("Duration can not be empty for simulator.")
%             end
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
%             % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ^ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % %%%%%%%%%%% ANTERIOR             |         %%%%%%%%%%%%%%%%%%%%%
%             % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% | %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             % % how many times did the prosthesis was moving suppossely.
%             % cs = obj.timing.c - obj.c0;
%             % duration = cs*obj.timing.period();
%             % obj.c0 = obj.timing.c;
%             % 
%             % if duration  == 0
%             % 
%             %     return;
%             % elseif duration < 0
%             %     warning("Duration lower than 0.")
%             %     return;
%             % elseif isempty(duration)
%             %     error("Duration can not be empty for simulator.")
%             % end
%             % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%             % %--------------- ANTES DESCOMENTADO -----------------------
%             % % trayectory before changing speed during this time
%             % trajectory = SimController.prosthesis_simulator( ...
%             %     obj.buffer(end, :), obj.vels, duration, ...
%             %     obj.sampling_period);
%             % %----------------------------------------------------------
%             %%%%%%%%%%%%%%%% ^    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%% |    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%% |    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             [trajectory, obj.phaseIdx, obj.phaseDir, obj.phaseSp] = SimController.prosthesis_simulator( ...
%             obj.buffer(end, :), obj.vels, duration, obj.sampling_period, ...
%             obj.phaseIdx, obj.phaseDir, obj.phaseSp);
%             %%%%%%%%%%%% 4 lineas comentadas %%%%%%%%%%%%
% 
% 
% 
% 
% 
%             obj.buffer = [obj.buffer; trajectory];
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%  PRINT PARA VER CS (TIMING)       %%%%%%%%%%%%%%%%
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             fprintf("timing.c=%d c0=%d cs=%d period=%.4f duration=%.4f\n", ...
%              obj.timing.c, obj.c0, cs, obj.timing.period(), duration);
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
%         end
%     end
% 
%     methods (Static)
%         trajectory = prosthesis_simulator( ...
%             initial_position, speeds, duration, sampling_period);
%     end
% end