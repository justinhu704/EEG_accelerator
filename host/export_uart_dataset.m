function export_uart_dataset(cacheFile, outDir)
%EXPORT_UART_DATASET Export all mixed-session test inputs for FPGA UART.
%
% Example:
%   export_uart_dataset( ...
%       "C:\Users\justi\OneDrive\文件\MATLAB\EEG_Hardware_preVerilog\EEG_105_stride10_mixedSess.mat", ...
%       "C:\EEG_Project\host\data");
%
% Output:
%   test_inputs_q12.bin  - sample-major, 3360 signed int16/sample,
%                          little-endian, Q12, floor + saturation
%   test_labels.csv      - sample_id, subject_label, fpga_label

    % Allow opening this function in MATLAB and pressing Run with no inputs.
    if nargin < 1 || strlength(string(cacheFile)) == 0
        cacheFile = fullfile(getenv('USERPROFILE'), 'OneDrive', '文件', ...
            'MATLAB', 'EEG_Hardware_preVerilog', ...
            'EEG_105_stride10_mixedSess.mat');
    end
    if nargin < 2 || strlength(string(outDir)) == 0
        outDir = fullfile(fileparts(mfilename('fullpath')), 'data');
    end
    cacheFile = string(cacheFile);
    outDir = string(outDir);

    if ~isfile(cacheFile)
        error("Dataset cache does not exist: %s", cacheFile);
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    % Use the standard file query first. Loading categorical objects through
    % matfile dot-indexing is less reliable than loading YTest explicitly.
    variableInfo = whos('-file', char(cacheFile));
    variableNames = string({variableInfo.name});
    if ~all(ismember(["XTest4", "YTest"], variableNames))
        error("The cache must contain XTest4 and YTest.");
    end

    source = matfile(char(cacheFile));
    inputSize = size(source, 'XTest4');
    if numel(inputSize) < 4
        inputSize(4) = 1;
    end
    if ~isequal(inputSize(1:3), [21 160 1])
        error("Expected XTest4 size 21x160x1xN, got %s.", ...
              mat2str(inputSize));
    end

    numSamples = inputSize(4);
    labelData = load(char(cacheFile), 'YTest');
    labels = labelData.YTest(:);
    if numel(labels) ~= numSamples
        error("XTest4 has %d samples but YTest has %d labels.", ...
              numSamples, numel(labels));
    end

    inputPath = fullfile(outDir, "test_inputs_q12.bin");
    expectedBytes = numSamples * 3360 * 2;
    existingInput = dir(char(inputPath));

    if ~isempty(existingInput) && existingInput.bytes == expectedBytes
        fprintf("Existing input binary is complete; skipping re-export.\n");
        fprintf("  %s (%d bytes)\n", inputPath, existingInput.bytes);
    else
        fileId = fopen(char(inputPath), 'wb', 'ieee-le');
        if fileId == -1
            error("Cannot open output file: %s", inputPath);
        end
        cleanupFile = onCleanup(@() fclose(fileId));

        % Reading one sample per matfile access is extremely slow for a 3 GB
        % HDF5 file, especially under OneDrive. Read and write large
        % contiguous batches instead.
        chunkSamples = 2048;
        fprintf("Exporting %d samples to %s\n", numSamples, inputPath);
        fprintf("Expected output: %.1f MiB, chunk size: %d samples\n", ...
                expectedBytes / 1024^2, chunkSamples);
        exportTimer = tic;

        for firstSample = 1:chunkSamples:numSamples
            lastSample = min(firstSample + chunkSamples - 1, numSamples);
            currentCount = lastSample - firstSample + 1;

            samples = source.XTest4(:,:,:,firstSample:lastSample);
            quantized = quantizeQ12Floor(samples);
            count = fwrite(fileId, quantized(:), "int16");
            expectedWords = 3360 * currentCount;
            if count ~= expectedWords
                error("Samples %d-%d wrote %d words instead of %d.", ...
                      firstSample, lastSample, count, expectedWords);
            end

            elapsedSeconds = toc(exportTimer);
            samplesPerSecond = lastSample / max(elapsedSeconds, eps);
            remainingSeconds = (numSamples-lastSample) / samplesPerSecond;
            fprintf("  %d / %d samples | %.1f samples/s | ETA %.1f min\n", ...
                    lastSample, numSamples, samplesPerSecond, ...
                    remainingSeconds/60);
        end
        clear cleanupFile; % Closes fileId through onCleanup.
    end

    % The labels are subject names such as S001, not numeric values. Map
    % them through the exact classNames saved with the trained network so
    % FPGA argmax index 0..104 matches MATLAB's FC output order. This also
    % handles skipped subjects S088/S092/S100/S104 correctly.
    modelPath = fullfile(fileparts(cacheFile), "EEG_CNN_GRU_mixedSess.mat");
    if isfile(modelPath)
        modelInfo = whos('-file', char(modelPath));
        modelVariables = string({modelInfo.name});
    else
        modelVariables = strings(0,1);
    end
    if ismember("classNames", modelVariables)
        modelData = load(char(modelPath), 'classNames');
        classNames = string(modelData.classNames(:));
        fprintf("Using class order from model: %s\n", modelPath);
    else
        classNames = string(categories(labels));
        fprintf("Warning: model classNames unavailable; using YTest categories.\n");
    end

    subjectLabel = string(labels);
    [labelFound, oneBasedClass] = ismember(subjectLabel, classNames);
    if ~all(labelFound)
        missingLabels = unique(subjectLabel(~labelFound));
        error("YTest contains labels absent from classNames: %s", ...
              strjoin(missingLabels, ", "));
    end
    if numel(classNames) ~= 105 || any(oneBasedClass < 1 | oneBasedClass > 105)
        error("Expected exactly 105 model classes, found %d.", numel(classNames));
    end

    sample_id = uint32((0:numSamples-1)');
    subject_label = subjectLabel;
    fpga_label = uint8(oneBasedClass - 1);
    labelTable = table(sample_id, subject_label, fpga_label);
    labelPath = fullfile(outDir, "test_labels.csv");
    writetable(labelTable, char(labelPath));

    fileInfo = dir(char(inputPath));
    if fileInfo.bytes ~= expectedBytes
        error("Binary size is %d bytes; expected %d.", ...
              fileInfo.bytes, expectedBytes);
    end

    fprintf("Finished.\n");
    fprintf("  Inputs: %s (%d bytes)\n", inputPath, fileInfo.bytes);
    fprintf("  Labels: %s (%d rows)\n", labelPath, numSamples);
end

function quantized = quantizeQ12Floor(values)
% Match EEG_Hardware_mixedSess.mlx QuantizeToInt for the input layer:
% total_bits=16, Fa_in=12, mode="floor".
    % Keep the calculation in single precision: multiplying a single by the
    % power-of-two scale 4096 is exact and avoids a large temporary double.
    integerCodes = floor(single(values) * single(4096));
    integerCodes = min(max(integerCodes, single(-32768)), single(32767));
    quantized = int16(integerCodes);
end
