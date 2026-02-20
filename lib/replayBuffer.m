classdef replayBuffer < handle
    properties
        capacity (1,1) double
        idx (1,1) double = 1
        count (1,1) double = 0
        obsDim (1,1) double

        states
        actions
        rewards
        nextStates
        dones
    end

    methods
        function this = replayBuffer(capacity, obsDim)
            this.capacity = capacity;
            this.obsDim = obsDim;

            this.states     = zeros(obsDim, capacity, 'single');
            this.nextStates = zeros(obsDim, capacity, 'single');
            this.actions    = zeros(1, capacity, 'uint16'); % store action index 1..A
            this.rewards    = zeros(1, capacity, 'single');
            this.dones      = false(1, capacity);
        end

        function add(this, s, aIdx, r, s2, done)
            this.states(:, this.idx) = single(s(:));
            this.actions(1, this.idx) = uint16(aIdx);
            this.rewards(1, this.idx) = single(r);
            this.nextStates(:, this.idx) = single(s2(:));
            this.dones(1, this.idx) = logical(done);

            this.idx = this.idx + 1;
            if this.idx > this.capacity
                this.idx = 1;
            end
            this.count = min(this.count + 1, this.capacity);
        end

        function [S, A, R, S2, D] = sample(this, batchSize)
            idxs = randi(this.count, [1 batchSize]);

            S  = this.states(:, idxs);
            A  = double(this.actions(:, idxs)); % [1,B]
            R  = this.rewards(:, idxs);
            S2 = this.nextStates(:, idxs);
            D  = this.dones(:, idxs);
        end
    end
end
