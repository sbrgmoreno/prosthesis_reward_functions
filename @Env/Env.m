classdef Env < rl.env.MATLABEnvironment
    %ENV class that handles the environment for reinforcement learning
    %{
    Laboratorio de Inteligencia y Visión Artificial
    ESCUELA POLITÉCNICA NACIONAL
    Quito - Ecuador
    
    autor: ztjona!
    jonathan.a.zea@ieee.org
    
    "I find that I don't understand things unless I try to program them."
    -Donald E. Knuth
    
    12 August 2021
    
    Mod after 2024/jan/4
    %}

    %% Properties
    % --- Hardware
    properties (AbortSet=true, GetAccess=public, SetAccess=protected, ...
            Transient=true)
        % AbortSet to avoid recreating object
        % Transient as session dependant
        glove (1,1) {isa(glove, 'Glove')}
        prosthesis (1, 1) {isa(prosthesis, 'Controller')}
        myo (1, 1) {isa(myo, 'Myo')}
    end

    %% Constants
    properties (Constant)
        % env
        v = 2.4; % must be changed in env changes.

        % --- Constants
        % when true, only 1 action. must match agent output.
        unifyActions = configurables('unifyActions');
        episodeDuration = configurables('episodeDuration'); % seconds
        speeds = configurables('speeds'); % vector of const speeds by motor
        period = configurables("period"); % reading period
        verbose = configurables('verbose'); % flag to print msgs

        returnHomeAtEndEpisode = false; % check!
        % % now returns to home if prerecorded.
        %returnHomeAtEndEpisode = configurables('returnHomeAtEndEpisode');

        % saves episode data
        flagSaveTraining = configurables("flagSaveTraining");
        episode_save_freq = configurables("episode_save_freq"); % every ith

        % NOTE: used only to initialize buffers
        maxNumberStepsInEpisodes=configurables("maxNumberStepsInEpisodes");

        % ---  property as method set in Configurables!
        % raw EMG -> NN input
        featureCalculator = configurables('fGetFeatures');

        % Scales raw encoder to glove range
        encoderNormCalculator = configurables("encoder2state_scale");
        
        % Scales flex data to [0 1]
        flexJoined_scaler = configurables("flexJoined_scale");

        % reward function
        reward_function = configurables("reward_function");

        % running simulator
        simMotors = configurables("simMotors");

        % clip actions
        rf_modify_actions = configurables("rf_modify_actions");
    end

    %% only in constructor
    properties (SetAccess=immutable)
        % --- using prerecorded
        usePrerecorded = true;
        emgSet = {}; % when using prerecordings
        gloveSet = {};
        sizeDataset = 0; % number of samples in dataset

        % -- saving
        episode_folder; % episode output folder
    end

    %% Properties that change during execution
    properties (SetAccess=protected)
        %---episode flags
        episodeType = ''; % EpisodeType.Opening or EpisodeType.Closing
        % Initialize system state
        State = [];

        % Initialize internal flags to indicate episode termination
        isDone = false; % true when episode finished

        c = 0; % steps in episode counter
        episodeCounter = 0;
        repetitionId = -1; % idx of the dataset repetition used in episode
        episodes_shuffled = []; % episodes in the dataset shuffled

        % now these use Timing class
        episodeTic; % timing of each episode
        periodTic;  % timing of each step inside episode

        % real tics-needed for half-hardware execution
        episode_realTic; % timing of each episode
        period_realTic;  % timing of each step inside episode

        % buffer main vars
        emg;
        motorData;
        flexData;
        emgLength;

        % buffers aux vars
        flexConverted;
        adjustEnc;

        % --- logs: for saving episode recording data
        % timestamp of init of episode with and without home [reset]
        episodeTimestamp = [0, 0];
        encoderLog = {};
        emgLog = {};
        encoderAdjustedLog = {};
        actionLog = [];%history of the actions per epidodes
        actionSatLog = [];%history of the actions per epidodes
        rewardLog = [];
        rewardIndividualLog = {};
        flexConvertedLog = {};

        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        % ---- Current-step metrics (NEW)
        % ---- Per-step metric logs (new)
        meanDistLog = [];   % [maxSteps x 1]
        mseLog = [];        % [maxSteps x 1]
        successLog = [];    % [maxSteps x 1]

        % ---- Per-episode metric logs (new)
        episodeCount = 0;        % contador para métricas agregadas (separado de episodeCounter si quieres)
        meanDistEpisode = [];    % [numEpisodes x 1]
        successRateEpisode = []; % [numEpisodes x 1]
        mseEpisode = [];         % [numEpisodes x 1]
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
                         %      METRICAS DE STEP    %
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        finalAbsErrEpisode = [];
        finalMeanAbsErr = NaN;
        finalMaxAbsErr = NaN;
        nearSuccessEpisode = NaN;
        nearSuccessLog = [];
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        

        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
                         %      METRICAS DE SENSIBILIDAD     %
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        % ---- Sensitivity logs (per-step)
        qLog = [];           % [maxSteps x 4] estado q (encoder->flex)
        qRefLog = [];        % [maxSteps x 4] referencia q_ref
        dqLog = [];          % [maxSteps x 4] delta q = q(t)-q(t-1)
        aRawLog = [];        % [maxSteps x 4] acción discreta antes de speeds
        aAppliedLog = [];    % [maxSteps x 4] acción aplicada (después de speeds)
        dirAgreeLog = [];    % [maxSteps x 4] bool: acción en dirección correcta
        effectNormLog = [];  % [maxSteps x 1] norm(dq)
        errNormLog = [];     % [maxSteps x 1] norm(q-q_ref)
        dErrLog = [];
        %-------------------------------------------------------------------%
        encRawLog = [];         % [maxSteps x 4] encoder crudo
        encEffectNormLog = [];  % [maxSteps x 1] norm(dEncoderCrudo)
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%

        wait_in_step = false; % bool to wait period
    end

    properties(GetAccess=public, SetAccess=public)
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        % ---- Current-step metrics (NEW)
        meanDistStep = NaN;    % escalar por step
        mseStep      = NaN;    % escalar por step
        successStep  = false;  % bool por step
        nearSuccessStep = false;
        

        forceActionDebug = false;          % OFF por defecto
        forcedActionValue = ones(1,4);     % valor cuando esté ON
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
    end




    methods
        %% Constructor
        % -----------------------------------------------------------------
        % Contructor method creates an instance of the environment
        function this = Env(agent_dir, usePrerecorded, emgs, gloveDatas)
            % # ---- Data Validation
            arguments
                agent_dir (1, 1) string = "";
                usePrerecorded	(1, 1) logical = false;
                emgs (:, :) cell = {};
                gloveDatas (:, :) cell = {};
            end

            % ------------ Initialize Observation settings
            ObservationInfo = Env.defineObservationInfo();

            % Initialize Action settings
            % ActionInfo = Env.defineActionInfo();
            
            
            ActionInfo = Env.defineActionDiscreteInfo();

            %%%%%% ^  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%% |  %%%%%% ACCIONES CONTINUAS 4 ACCIONES [-1, 1] %%%%%%%%%
            %%%%%% |  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %ActionInfo = Env.defineActionInfo();
            %%%%%%%%%%%%%%%%%% 1 LINEA COMENTADA %%%%%%%%%%%%%%%%%%%%%%%%%%
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(...
                ObservationInfo, ActionInfo);

            this.log("Defined observation and action space");

            % --- inmutable properties
            this.episode_folder = agent_dir;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%% CUANDO NO HAY CARPETA DONDE EJECUTAR %%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % --- episode saving folder (robust default)
            if strlength(agent_dir) == 0
                this.episode_folder = string(fullfile(getenv("USERPROFILE"), "Documents", "MATLAB", "prosthesis_episodes"));
            else
                this.episode_folder = agent_dir;
            end
            if ~exist(this.episode_folder, "dir")
                mkdir(this.episode_folder);
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % --- hardware
            this.usePrerecorded = usePrerecorded;

            if usePrerecorded
                this.log("Using prerecorded data");
                this.emgSet = emgs;
                this.gloveSet = gloveDatas;
                this.sizeDataset = size(emgs, 1);

                % loading with a random episode (i.e. 1)
                this.myo = RecordedMyo(emgs{1});
                this.glove = RecordedGlove(gloveDatas{1});
            else
                this.log("Connecting to devices");

               this.myo = Myo();
               % this.myo =RecordedMyo(emgs{1});
                if configurables("connect_glove")
                    % real glove
                    this.glove = Glove(configurables("comGlove"));
                else
                    % overload glove
                    this.glove = FakeGlove();
                end

                this.log("Created devices");
            end

            % -- simulation
            if true %false %cambiar a true valor de la bandera no lee para evaluar cambiar a false

                % when input false it is in simulation
                this.episodeTic = Timing(false, this.period);
                this.periodTic = Timing(false, this.period);

                this.prosthesis = SimController(this.episodeTic);
            else
                % hardware
                % when input true it is in real world hardware
                this.episodeTic = Timing(true);
                this.periodTic = Timing(true);
                disp('Env Hardware')
%                com = configurables('comUNO');
                 com = "COM6";

                msg = "Connecting to serial port %s\n" + ...
                    "If required," + ...
                    "change the prosthesis port in config\\configurables.m";
                this.log(sprintf( msg, com));
                this.prosthesis = Controller(false, '', com);
            end

            this.wait_in_step = ~this.simMotors || ~this.usePrerecorded;

            if this.wait_in_step
                this.log("Waiting in step");
                this.period_realTic = tic;
                this.episode_realTic = tic;
            else
                this.period_realTic = [];
                this.episode_realTic = [];
            end

            %-----
            % Initialize property values and pre-compute necessary values
            % updateActionInfo(this); % actions always the same

            % validate env
            % this.log("Validating env");
            % this.validateEnvironment();
            % this.log("Validated!");
            
            %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
            % ---- Episodic metrics (init once)
            this.episodeCount = 0;
            this.meanDistEpisode = [];
            this.successRateEpisode = [];
            this.mseEpisode = [];

            % ---- Per-step metric logs (init empty; reset() will size them)
            this.meanDistLog = [];
            this.mseLog = [];
            this.successLog = [];
            %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%----------------------%%%%%%%%%%%%%%%%%%%%%%%%%

            this.log("Env. created.");
        end
    end

    %% methods
    % DOCS:
    % [1] https://www.mathworks.com/help/reinforcement-learning/ug/
    % create-custom-matlab-environment-from-template.html
    %
    % Helper methods to create the environment
    % -------- From the docs [1]:
    % The getObservationInfo, getActionInfo, sim, and
    % validateEnvironment functions are already defined in the base
    % abstract class.

    % getObservationInfo(this)
    % actionInfo = getActionInfo(this)
    % validateEnvironment = validateEnvironment(this)
    %sim = sim(this, agente);


    % To create your environment, you must define the constructor,
    % reset, and step functions.
    methods
        InitialObservation = reset(this)
        [Observation,Reward,IsDone,LoggedSignals] = step(this, action)
        state = calculateState(this, emg, motorData)

        isDone = checkEndEpisode(this)

        loop(this, agent) %

        saveEpisode(this)

        plot_episode(this)

        plot_episode2(this)

        % -----------------------------------------------------------------
        function log(this, msg)
            % prints messages depending on verbose flag.
            if this.verbose
                fprintf('%s|| %s\n', string(datetime("now", "Format", ...
                    'yy-MM-dd HH:m:ss.SSS')), msg);
            end
        end

    end

    %% Other methods
    % /////////////////////////////////////////////////////////////////////
    methods (Access = protected)
        % (optional) update visualization everytime the environment is
        % updated
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end

    % /////////////////////////////////////////////////////////////////////
    methods (Static)

        obsInfo = defineObservationInfo()

        actionInfo = defineActionDiscreteInfo()

        %--- continous version
        actionInfo = defineActionInfo() %antes comentado

    end
end
% More properties at: AbortSet, Abstract, Access, Dependent, GetAccess, ...
% GetObservable, NonCopyable, PartialMatchPriority, SetAccess, ...
% SetObservable, Transient, Framework attributes
% https://www.mathworks.com/help/matlab/matlab_oop/property-attributes.html

% Methods: Abstract, Access, Hidden, Sealed, Framework attributes
% https://www.mathworks.com/help/matlab/matlab_oop/method-attributes.html

%https://www.mathworks.com/help/reinforcement-learning/ug/
% create-custom-matlab-environment-from-template.html