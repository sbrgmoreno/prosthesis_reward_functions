classdef DuelingCombineLayer < nnet.layer.Layer
    % Combina V(s) y A(s,a) -> Q(s,a)
    % Entrada: X = [V; A] con tamaño (1+numActions) x batch
    % Salida: Q con tamaño numActions x batch

    properties
        NumActions
    end

    methods
        function layer = DuelingCombineLayer(numActions, name)
            layer.Name = name;
            layer.Description = "Dueling combine: Q = V + (A - mean(A))";
            layer.NumActions = numActions;
        end

        function Z = predict(layer, X)
            V = X(1, :);           % 1 x batch
            A = X(2:end, :);       % numActions x batch
            Amean = mean(A, 1);    % 1 x batch
            Z = V + (A - Amean);   % numActions x batch
        end
    end
end

