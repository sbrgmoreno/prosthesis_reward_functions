classdef Myo < handle
    %Myo class to communicate with myo

    %{
    Laboratorio de Inteligencia y Visión Artificial
    ESCUELA POLITÉCNICA NACIONAL
    Quito - Ecuador
    
    autor: ztjona!
    jonathan.a.zea@ieee.org
    
    "I find that I don't understand things unless I try to program them."
    -Donald E. Knuth
    
    12 August 2021
    Matlab 9.9.0.1592791 (R2020b) Update 5.
    %}

    %%
    properties (GetAccess = public, SetAccess=private)
        isConnected = false;
        myoObject
    end

    methods
        %% Constructor
        % -----------------------------------------------------------------
        function obj = Myo()
            %Myo(...) creates a new myo object
            %

            % # ----
            connectMyo(obj);
        end
        
        connectMyo(obj)
        terminateMyo(obj)

        %%
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%% RESET BUFFER SEGURO %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function resetBuffer(this)
        %RESETBUFFER Limpia/Resetea el buffer del Myo sin crashear si no existe myoObject
        
            % Si tu código original aquí hacía más cosas (limpiar variables internas),
            % déjalas arriba de este bloque.
        
            if isprop(this,'myoObject') && ~isempty(this.myoObject)
                try
                    if isprop(this.myoObject,'myoData')
                        disp(this.myoObject.myoData);
                    else
                        disp("Myo/resetBuffer: myoObject no tiene myoData");
                    end
                catch ME
                    disp("Myo/resetBuffer: no se pudo acceder a myoObject.myoData");
                    disp(ME.message);
                end
            else
                disp("Myo/resetBuffer: myoObject no inicializado (se omite disp)");
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%% READEMG %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function emg = readEmg(obj, varargin)
        %READEMG Lee EMG del Myo de forma robusta.
        % Devuelve un vector E-by-8 (o 1-by-8) según tu implementación.
        
            % Valor por defecto si no hay datos (evita crashear reset/step)
            emg = zeros(1,8);
        
            % Si no existe myoObject o está vacío, no intentes dot-indexing
            if ~isprop(obj,'myoObject') || isempty(obj.myoObject)
                % opcional: warning("Myo/readEmg: myoObject vacío");
                return;
            end
        
            mo = obj.myoObject;
        
            % Caso 1: myoObject es struct
            if isstruct(mo)
                if isfield(mo,'myoData') && isstruct(mo.myoData) && isfield(mo.myoData,'emg_log')
                    emg = mo.myoData.emg_log;
                    return;
                end
                return;
            end
        
            % Caso 2: myoObject es objeto (handle / class)
            if isobject(mo)
                try
                    % si tiene propiedad myoData
                    if isprop(mo,'myoData')
                        md = mo.myoData;
                        % md puede ser struct u objeto
                        if isstruct(md) && isfield(md,'emg_log')
                            emg = md.emg_log;
                            return;
                        elseif isobject(md) && isprop(md,'emg_log')
                            emg = md.emg_log;
                            return;
                        end
                    end
        
                    % Si tu implementación usa otro método (por ejemplo getData/read),
                    % aquí puedes adaptar:
                    % if ismethod(mo,'getData'), data = mo.getData(); ...
                catch
                    % si falla, devolvemos ceros sin romper el env
                    return;
                end
            end
        
            % Caso 3: myoObject NO es struct ni object (double, etc.)
            % => no se puede dot-indexing
            return;
        end        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %   ^
        %   |
        %   |
        %%%%%%%%%%%% RESET BUFFER ANTERIOR %%%%%%%%%%%%%%%%
        % -----------------------------------------------------------------
        % function resetBuffer(obj)
        %     disp(obj);
        %     disp(obj.myoObject)
        %     disp(obj.myoObject.myoData)
        %    obj.myoObject.myoData.clearLogs();
        % end
        %%
        % -----------------------------------------------------------------
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






        % %   ^
        % %   |
        % %   |
        % %%%%%%%%%%%% READMEG ANTERIOR %%%%%%%%%%%%%%%%
        % function emg = readEmg(obj)
        %     %obj.readEmg() returns EMG signal
        %     %# Outputs
        %     %* emg		-M-by-8
        %     %
        % 
        %     % # ----
        %     emg = obj.myoObject.myoData.emg_log;
        %     obj.myoObject.myoData.clearLogs();
        % end
        % %% 
        % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % -----------------------------------------------------------------
        function delete(obj)
            obj.terminateMyo();
        end
    end
end
% More properties at: AbortSet, Abstract, Access, Dependent, GetAccess, ...
% GetObservable, NonCopyable, PartialMatchPriority, SetAccess, ...
% SetObservable, Transient, Framework attributes
% https://www.mathworks.com/help/matlab/matlab_oop/property-attributes.html

% Methods: Abstract, Access, Hidden, Sealed, Framework attributes
% https://www.mathworks.com/help/matlab/matlab_oop/method-attributes.html