%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% 81  y 9 y [17 (hibridos)] acciones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function actionInfo = defineActionDiscreteInfo()
% defineActionDiscreteInfo() defines discrete actions for the environment.
%
% Modes:
% - unifyActions = true          -> scalar action {-1,0,1}
% - actionMode = "full_81"       -> 81 joint actions
% - actionMode = "single_motor_9"-> 9 atomic actions
% - actionMode = "hybrid_17"     -> 17 actions (9 atomic + 8 coordinated)
%
% ============================================================

%% config
if configurables('unifyActions')
    actionInfo = rlFiniteSetSpec([-1 0 1]);

else
    actionMode = configurables('actionMode');

    switch string(actionMode)

        case "full_81"
            vals = [-1 0 1];
            actions = [];
            for a = vals
                for b = vals
                    for c = vals
                        for d = vals
                            actions = [actions; a b c d];
                        end
                    end
                end
            end
            actionInfo = rlFiniteSetSpec(num2cell(actions,2)');

        case "single_motor_9"
            actions = [ ...
                 0  0  0  0;   % no-op
                 1  0  0  0;   % m1 +
                -1  0  0  0;   % m1 -
                 0  1  0  0;   % m2 +
                 0 -1  0  0;   % m2 -
                 0  0  1  0;   % m3 +
                 0  0 -1  0;   % m3 -
                 0  0  0  1;   % m4 +
                 0  0  0 -1];  % m4 -

            actionInfo = rlFiniteSetSpec(num2cell(actions,2)');

        case "hybrid_17"
            actions = [ ...
                % -------- atomic actions (9) --------
                 0  0  0  0;   % no-op
                 1  0  0  0;   % m1 +
                -1  0  0  0;   % m1 -
                 0  1  0  0;   % m2 +
                 0 -1  0  0;   % m2 -
                 0  0  1  0;   % m3 +
                 0  0 -1  0;   % m3 -
                 0  0  0  1;   % m4 +
                 0  0  0 -1;   % m4 -

                % ----- coordinated actions (8) -----
                 1  1  0  0;   % m1,m2 +
                -1 -1  0  0;   % m1,m2 -
                 0  0  1  1;   % m3,m4 +
                 0  0 -1 -1;   % m3,m4 -
                 1  0  1  0;   % m1,m3 +
                -1  0 -1  0;   % m1,m3 -
                 0  1  0  1;   % m2,m4 +
                 0 -1  0 -1];  % m2,m4 -

            actionInfo = rlFiniteSetSpec(num2cell(actions,2)');

        otherwise
            error("defineActionDiscreteInfo: Unknown actionMode '%s'", string(actionMode));
    end
end

%% properties
actionInfo.Name = 'prosthesis_action_space';
actionInfo.Description = 'Discrete action space for prosthesis control.';
end








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% 81 acciones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function actionInfo = defineActionDiscreteInfo()
% 
% if configurables('unifyActions')
%     actionInfo = rlFiniteSetSpec([-1 0 1]);
% else
%     vals = [-1 0 1];
%     actions = [];
%     for a = vals
%         for b = vals
%             for c = vals
%                 for d = vals
%                     actions = [actions; a b c d];
%                 end
%             end
%         end
%     end
%     actionInfo = rlFiniteSetSpec(num2cell(actions,2)');
% end
% 
% actionInfo.Name = 'prosthesis_action_space';
% actionInfo.Description = ...
%     'Discrete action space with 81 combinations for 4 motors.';
% end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% 9 acciones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function actionInfo = defineActionDiscreteInfo()
% % defineActionDiscreteInfo() defines a reduced discrete action space:
% % one motor moves per step, or no movement.
% 
% if configurables('unifyActions')
%     actionInfo = rlFiniteSetSpec([-1 0 1]);
% else
%     actions = { ...
%         [ 1  0  0  0], ...  % motor 1 open
%         [-1  0  0  0], ...  % motor 1 close
%         [ 0  1  0  0], ...  % motor 2 open
%         [ 0 -1  0  0], ...  % motor 2 close
%         [ 0  0  1  0], ...  % motor 3 open
%         [ 0  0 -1  0], ...  % motor 3 close
%         [ 0  0  0  1], ...  % motor 4 open
%         [ 0  0  0 -1], ...  % motor 4 close
%         [ 0  0  0  0]  ...  % stop
%     };
% 
%     actionInfo = rlFiniteSetSpec(actions);
% end
% 
% actionInfo.Name = 'prosthesis_action_space';
% actionInfo.Description = ...
%     'Reduced action space: one motor per step or stop.';
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




% function actionInfo = defineActionDiscreteInfo()
% % defineActionDiscreteInfo() defines discrete actions, currently, only 3
% % actions for each motor of the agent
% %
% %
% 
% %{
% Laboratorio de Inteligencia y Visión Artificial
% ESCUELA POLITÉCNICA NACIONAL
% Quito - Ecuador
% 
% autor: ztjona
% jonathan.a.zea@ieee.org
% Cuando escribí este código, solo dios y yo sabíamos como funcionaba.
% Ahora solo lo sabe dios.
% 
% "I find that I don't understand things unless I try to program them."
% -Donald E. Knuth
% 
% 31 January 2022
% 
% Mod 2024/jan/3
% %}
% 
% %% configs
% % each motor forward, stop or backward.
% %
% if configurables('unifyActions')
%     actionInfo = rlFiniteSetSpec([-1 0 1]);
% else
%     % % Generates 81 actions, the
%     % % combinations of the 4 motors and 3 actions
%     actions = combvec([-1 0 1], [-1 0 1], [-1 0 1], [-1 0 1])';    
%     actionInfo = rlFiniteSetSpec(num2cell(actions, 2)');
% 
%   % actions = {[-1 -1 -1 -1],[0 0 0 0],[1 1 1 1]};
%   % actionInfo = rlFiniteSetSpec(actions);
% end
% %% defining properties
% 
% actionInfo.Name = 'prosthesis_action_space';
% actionInfo.Description = ...
%     'Actions defined as forward, stop and backward for all motors.';