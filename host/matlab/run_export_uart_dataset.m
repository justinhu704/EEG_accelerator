% One-click launcher for exporting the complete mixed-session test set.
% Open this file in MATLAB and press Run.

scriptDir = fileparts(mfilename('fullpath'));
addpath(scriptDir);

if ~exist('datasetFile', 'var') || strlength(string(datasetFile)) == 0
    datasetFile = fullfile(getenv('USERPROFILE'), 'OneDrive', '文件', ...
        'MATLAB', 'EEG_Hardware_preVerilog', ...
        'EEG_105_stride10_mixedSess.mat');
end
if ~exist('outputFolder', 'var') || strlength(string(outputFolder)) == 0
    outputFolder = fullfile(fileparts(scriptDir), 'data');
end

fprintf('Dataset: %s\n', datasetFile);
fprintf('Output : %s\n', outputFolder);

if ~isfile(datasetFile)
    error('Dataset file was not found: %s. Set datasetFile to the actual cache path before running this script.', datasetFile);
end

export_uart_dataset(datasetFile, outputFolder);
