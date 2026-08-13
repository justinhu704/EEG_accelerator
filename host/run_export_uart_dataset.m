% One-click launcher for exporting the complete mixed-session test set.
% Open this file in MATLAB and press Run.

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

datasetFile = fullfile(getenv('USERPROFILE'), 'OneDrive', '文件', ...
    'MATLAB', 'EEG_Hardware_preVerilog', ...
    'EEG_105_stride10_mixedSess.mat');
outputFolder = fullfile(scriptDir, 'data');

fprintf('Dataset: %s\n', datasetFile);
fprintf('Output : %s\n', outputFolder);

if ~isfile(datasetFile)
    error('Dataset file was not found: %s', datasetFile);
end

export_uart_dataset(datasetFile, outputFolder);
