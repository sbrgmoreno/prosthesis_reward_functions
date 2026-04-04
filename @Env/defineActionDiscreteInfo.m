%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% 81  y 9 y [17 (hibridos)] acciones %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function actionInfo = defineActionDiscreteInfo()
% defineActionDiscreteInfo() defines discrete actions for the environment.
%
% Modes:
% - unifyActions = true            -> scalar action {-1,0,1}
% - actionMode = "full_81"         -> 81 joint actions
% - actionMode = "single_motor_9"  -> 9 atomic actions
% - actionMode = "hybrid_17"       -> 17 actions (9 atomic + 8 coordinated)
% - actionMode = "structured_25"   -> 25 structured actions
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

        case "structured_23"
            actions = [ ...
                % ===== Group A: neutral =====
                 0  0  0  0;   %  1 no-op

                % ===== Group B: single-motor actions (8) =====
                 1  0  0  0;   %  2 m1 +
                -1  0  0  0;   %  3 m1 -
                 0  1  0  0;   %  4 m2 +
                 0 -1  0  0;   %  5 m2 -
                 0  0  1  0;   %  6 m3 +
                 0  0 -1  0;   %  7 m3 -
                 0  0  0  1;   %  8 m4 +
                 0  0  0 -1;   %  9 m4 -

                % ===== Group C: coordinated pairs already suggested by prior runs (8) =====
                 1  1  0  0;   % 10 m1,m2 +
                -1 -1  0  0;   % 11 m1,m2 -
                 1  0  1  0;   % 12 m1,m3 +
                -1  0 -1  0;   % 13 m1,m3 -
                 0  1  0  1;   % 14 m2,m4 +
                 0 -1  0 -1;   % 15 m2,m4 -
                 0  0  1  1;   % 16 m3,m4 +
                 0  0 -1 -1;   % 17 m3,m4 -

                % ===== Group D: diagonal/alternative coordinated pairs (4) =====
                 1  0  0  1;   % 18 m1,m4 +
                -1  0  0 -1;   % 19 m1,m4 -
                 0  1  1  0;   % 20 m2,m3 +
                 0 -1 -1  0;   % 21 m2,m3 -

                % ===== Group E: global/coarse coordinated actions (4) =====               
                 1  1 -1 -1;   % 22 split pattern A
                -1 -1  1  1];  % 23 split pattern B

            actionInfo = rlFiniteSetSpec(num2cell(actions,2)');

        case "structured_25"
            actions = [ ...
                % ===== Group A: neutral =====
                 0  0  0  0;   %  1 no-op

                % ===== Group B: single-motor actions (8) =====
                 1  0  0  0;   %  2 m1 +
                -1  0  0  0;   %  3 m1 -
                 0  1  0  0;   %  4 m2 +
                 0 -1  0  0;   %  5 m2 -
                 0  0  1  0;   %  6 m3 +
                 0  0 -1  0;   %  7 m3 -
                 0  0  0  1;   %  8 m4 +
                 0  0  0 -1;   %  9 m4 -

                % ===== Group C: coordinated pairs already suggested by prior runs (8) =====
                 1  1  0  0;   % 10 m1,m2 +
                -1 -1  0  0;   % 11 m1,m2 -
                 1  0  1  0;   % 12 m1,m3 +
                -1  0 -1  0;   % 13 m1,m3 -
                 0  1  0  1;   % 14 m2,m4 +
                 0 -1  0 -1;   % 15 m2,m4 -
                 0  0  1  1;   % 16 m3,m4 +
                 0  0 -1 -1;   % 17 m3,m4 -

                % ===== Group D: diagonal/alternative coordinated pairs (4) =====
                 1  0  0  1;   % 18 m1,m4 +
                -1  0  0 -1;   % 19 m1,m4 -
                 0  1  1  0;   % 20 m2,m3 +
                 0 -1 -1  0;   % 21 m2,m3 -

                % ===== Group E: global/coarse coordinated actions (4) =====
                 1  1  1  1;   % 22 all +
                -1 -1 -1 -1;   % 23 all -
                 1  1 -1 -1;   % 24 split pattern A
                -1 -1  1  1];  % 25 split pattern B

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