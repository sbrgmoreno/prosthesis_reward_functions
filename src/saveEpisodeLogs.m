function saveEpisodeLogs(this)
% Save current episode logs to a .mat file

    % ===== folder =====
    if ~isprop(this, 'episode_folder') || isempty(this.episode_folder)
        warning('saveEpisodeLogs: episode_folder is empty. Episode not saved.');
        return;
    end

    if ~exist(this.episode_folder, 'dir')
        mkdir(this.episode_folder);
    end

    % ===== episode number =====
    if isprop(this, 'episodeCount')
        episodeNumber = this.episodeCount;
    elseif isprop(this, 'episode')
        episodeNumber = this.episode;
    else
        episodeNumber = 0;
    end

    % ===== safe extraction =====
    data = struct();

    data.episodeNumber = episodeNumber;

    if isprop(this, 'aRawLog'),        data.aRawLog = this.aRawLog; end
    if isprop(this, 'actionSatLog'),   data.actionSatLog = this.actionSatLog; end
    if isprop(this, 'qLog'),           data.qLog = this.qLog; end
    if isprop(this, 'qRefLog'),        data.qRefLog = this.qRefLog; end
    if isprop(this, 'dqLog'),          data.dqLog = this.dqLog; end
    if isprop(this, 'rewardLog'),      data.rewardLog = this.rewardLog; end
    if isprop(this, 'effectNormLog'),  data.effectNormLog = this.effectNormLog; end
    if isprop(this, 'dirAgreeLog'),    data.dirAgreeLog = this.dirAgreeLog; end
    if isprop(this, 'dErrLog'),        data.dErrLog = this.dErrLog; end
    if isprop(this, 'meanDistLog'),    data.meanDistLog = this.meanDistLog; end
    if isprop(this, 'mseLog'),         data.mseLog = this.mseLog; end
    if isprop(this, 'successLog'),     data.successLog = this.successLog; end

    % ===== optional metadata =====
    if isprop(this, 'period'),         data.period = this.period; end
    if isprop(this, 'actionGain'),     data.actionGain = this.actionGain; end
    if isprop(this, 'actionMode'),     data.actionMode = this.actionMode; end
    if isprop(this, 'rewardType'),     data.rewardType = this.rewardType; end

    % ===== summary values =====
    if isfield(data, 'rewardLog') && ~isempty(data.rewardLog)
        data.rewardSum = sum(data.rewardLog);
    else
        data.rewardSum = NaN;
    end

    if isfield(data, 'aRawLog') && ~isempty(data.aRawLog)
        actionList = unique(data.aRawLog, 'rows');
        data.actionList = actionList;
        data.numUniqueActions = size(actionList, 1);
    else
        data.actionList = [];
        data.numUniqueActions = 0;
    end

    % ===== filename =====
    fileName = fullfile(this.episode_folder, sprintf('episode_%04d.mat', episodeNumber));

    save(fileName, '-struct', 'data');
end
