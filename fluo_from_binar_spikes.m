clear all
clc
close all

% Define specific parameters (update these as needed)
topology = 'Flat';
dynamics = 'RS'; 
L = 0.1; 
ei = 0.8; 
no = 0.05; 
sigma = 1.0; % fluo noise
dt = 1e-3; 
Texp = 30000; % Experiment time

% Samples
samplesize = 50; % Number of random neurons from the culture

% Calcium fluorescence parameters
Type = 2; % 'GECI'
P2 = 0.81;
P3 = -0.056;
F0 = 42; 
tau = 1.82;
A = 11.3 / 100;  

% Folder to load and save data
FolderName = './RESULTS/';

% spike and roi files to loas
spikename = sprintf('%s_%s_L_%.1f_spikes.txt', topology, dynamics, L);
roiname = sprintf('%s_%s_L_%.1f_ROIs.txt', topology, dynamics, L);

%%
file_name_spk = fullfile(FolderName, spikename);
file_name_roi = fullfile(FolderName, roiname);

spikes = readtable(file_name_spk);
ROI = readtable(file_name_roi);

% Process data
texp = Texp / 1000; % Convert time unit to seconds
time = 0:dt:texp;

disp(['Dynamics: ', dynamics, ', L = ', num2str(L), ', Iz-noise = ', num2str(no), ', dt = ', num2str(dt), ', sigma = ', num2str(sigma)]);

[subspikes, subROI] = selectSubset(spikes, ROI, samplesize, true);

subspikes{:, 2} = subspikes{:, 2} / 1000; % Convert spike-time unit to seconds
subspikes.Properties.VariableNames = {'ROI_id', 'time'}; % Define column names

% Save spikes and ROI files
disp('Save spikes and ROI files.')
outputSubspikes = sprintf('%s_%s_L_%.1f_spikes.txt', topology, dynamics, L);
outputSubrois = sprintf('%s_%s_L_%.1f_rois.txt', topology, dynamics, L);
writetable(subspikes, fullfile(FolderName, outputSubspikes));
writetable(subROI, fullfile(FolderName, outputSubrois));
disp('Done.');

% Calculate fluorescence from spikes
disp('Now calculate fluorescence...');
fluorp = fluo_traces(subspikes, A, F0, tau, P2, P3, sigma, dt, texp, Type);

% Save fluorescence
disp('Save fluo traces.');
outputfluorp = sprintf('%s_%s_L_%.1f_calcium.txt', topology, dynamics, L);         
fluorpFileName = fullfile(FolderName, outputfluorp);

% Save fluorescence data to text file
dlmwrite(fluorpFileName, fluorp, 'delimiter', '\t', 'precision', '%.7f');

% Compress the text file to .gz
gzip(fluorpFileName);

% Delete the original text file (optional)
delete(fluorpFileName);

disp('Done.');

% Plot some calcium traces
disp('Plot some calcium traces.')
sb = 5;
selectedSpikes = selectTrains(subspikes, subROI, sb);

plt = figure();
for j = 1:sb
    subplot(sb, 1, j);
    train_j = selectedSpikes{j, :};
    spikeTimes = train_j{1}(2:end);
    train = createSpikeTrain(spikeTimes, texp, dt);

    % Plot fluorp
    plot(fluorp(j, :), 'b');
    hold on;

    % Plot train
    plot(train * (max(fluorp(j, :), [], 'all') + max(fluorp(j, :), [], 'all') / 20), 'or', 'MarkerSize', 3, 'MarkerFaceColor', 'r');
    ylim([min(fluorp(j, :)) - min(fluorp(j, :)) / 10, max(fluorp(j, :)) + max(fluorp(j, :)) / 10]);
    xlim([0, texp / dt]);
    yticks([F0, max(fluorp(j, :))]);
    yticklabels({sprintf('%.2f', F0), sprintf('%.2f', max(fluorp(j, :)))});
    y_range = get(gca, 'YLim');
    y_midpoint = mean(y_range);
    text(1.04 * length(fluorp(j, :)), y_midpoint, 'F(t)', 'Rotation', 0, 'HorizontalAlignment', 'right');

    % Add the "roi=k" annotation
    k = subROI{j, 1};
    text(0.05 * length(time), 1.05 * max(fluorp(j, :)), sprintf('roi=%d', k), 'FontSize', 10, 'Color', 'k');

    % Add x-axis label only in the bottom plot
    if j == sb
        xlabel('time (ms)');
    else
        set(gca, 'xtick', []);
        set(gca, 'xticklabel', []);
    end

    hold off;
end

sgtitle(sprintf('Ca2+ fluo: %s, %s, L=%.1g, E/I=%1g, Iz noise=%2g, Ca2+ noise=%g', topology, dynamics, L, ei, no, sigma));
figFilename = fullfile(FolderName, sprintf('calcium_traces.png'));
saveas(gcf, figFilename);
disp('Done.');
close(plt);


%% *FUNCTIONS*



function [out] = fluo_traces(spikes, A, F0, tau, P2, P3, sigma, dt, texp, Type)
    % This function converts the spikes data (in netcal format) into
    % fluorescence traces based on a reparameterized physiological model for
    % synthetic calcium indicators and a cubic polynomial representation for
    % genetically encoded ones.

    % Convert parameters to single precision
    A = single(A);
    F0 = single(F0);
    tau = single(tau);
    P2 = single(P2);
    P3 = single(P3);
    sigma = single(sigma);
    dt = single(dt);
    texp = single(texp);

    % Collect the firing times for each ROI
    unique_indices = unique(spikes{:, 1}); % Find all unique integer values in the first column
    num_rois = length(unique_indices);

    spktimes = cell(num_rois, 1);
    for i = 1:num_rois
        idxN = spikes{:, 1} == unique_indices(i);
        spktimes{i, 1} = single(spikes{idxN, 2}); % Convert to single precision
    end

    % Initiating variables
    t = single(0:dt:texp); 
    num_timepoints = length(t);

    % Initialize fluorescence matrix with single precision
    fluo = ones(num_rois, num_timepoints, 'single') * F0;

    % Such addition allows to get noises different for each ROI
    for i = 1:num_rois 
        white_n = generateWhiteNoise(sigma, num_timepoints);
        fluo(i, :) = fluo(i, :) + white_n;
    end

    if Type == 1
        for j = 1:num_rois  % Governs the ROI's choice 
            for i = 1:length(spktimes{j, 1})
                ispk = round(spktimes{j, 1}(i) / dt);
                ispk = ispk + 1;  % Convert zero-based indices to one-based indices (Python)
                if ispk <= num_timepoints
                    fluo(j, ispk+1:end) = fluo(j, ispk+1:end) + F0 * A * exp(-(t(ispk+1:end) - t(ispk)) / tau);
                end
            end
        end
    else
        for j = 1:num_rois  % Governs the ROI's choice 
            for i = 1:length(spktimes{j, 1})
                ispk = round(spktimes{j, 1}(i) / dt);
                ispk = ispk + 1;  % Convert zero-based indices to one-based indices (Python)
                if ispk <= num_timepoints
                    exp_term = exp(-(t(ispk+1:end) - t(ispk)) / tau);
                    fluo(j, ispk+1:end) = fluo(j, ispk+1:end) + F0 * A * (exp_term + ...
                        P2 * ((exp_term .^ 2) - exp_term) + ...
                        P3 * ((exp_term .^ 3) - exp_term));
                end
            end
        end
    end

    out = fluo;
end


function white_n = generateWhiteNoise(sigma, len_t)
    % Generate white noise in single precision
    white_n = single(sigma) * randn(1, len_t, 'single'); 
end



%%
function [subspikes, subROI] = selectSubset(spikes, ROI, M, shuffle)
    % M = size of the ROI subset for sampling
    % shuffle = boolean indicating whether to shuffle the indices

    % Check if M is provided
    if nargin > 2
        % Get unique indices from spikes
        unique_indices = unique(spikes{:, 1});
        
        % Shuffle the unique indices if shuffle is true
        if shuffle
            unique_indices = unique_indices(randperm(length(unique_indices)));
        end
        
        % Select first M unique indices
        subidx = unique_indices(1:min(length(unique_indices), M));

        % Select rows from spikes where the index is in subidx
        subspikes = spikes(ismember(spikes{:, 1}, subidx), :);

        % Select subset of ROI corresponding to subidx
        subROI = ROI(ismember(ROI{:, 1}, subidx), :);
    else
        % No subsampling, return spikes and ROI as is
        subspikes = spikes;
        subROI = ROI;
    end
end


function selectedSubset = selectTrains(subspikes, subROI, N)
    % Initialize a cell array to store selected values
    selectedSubset = cell(N, 1);
    
    % Iterate through each of the N elements
    for i = 1:N
        % Get the current element from the ith row of subROI
        currentElement = subROI{i, 1};
        
        % Find all occurrences of the current element in the first column of subspikes
        occurrences = find(subspikes{:, 1} == currentElement);
        
        % Initialize an array to store corresponding values from subspikes
        values = zeros(length(occurrences), 1);
        
        % Extract the values corresponding to the occurrences
        for j = 1:length(occurrences)
            values(j) = subspikes{occurrences(j), 2};
        end
        
        % Store the values in the cell array along with the current element
        selectedSubset{i} = [currentElement; values];
    end
    
    % Convert the cell array to a table
    selectedSubset = cell2table(selectedSubset);
end

function result = createSpikeTrain(spikelist, maxTime, dt)
    % Calculate the number of elements in the time list
    numElements = ceil(maxTime / dt);

    % Initialize the result list with zeros
    result = zeros(1, numElements);

    % Calculate the indices corresponding to spikelist in the result list
    indices = ceil(spikelist / dt);

    % Set the values at the indices corresponding to spikelist to 1
    indices = indices + 1; % Convert zero-based to one-based (python)
    result(indices) = 1;
end


