function report = EEG_Quantize(numSamples, outputFolder, exportGoldenFiles)
%EEG_QUANTIZE Export and verify the trained EEG model without retraining.
%
%   report = EEG_Quantize()
%   report = EEG_Quantize(4)
%   report = EEG_Quantize(100, "C:\path\to\output")
%   report = EEG_Quantize(100, "", false)
%   report = EEG_Quantize(inf)  % all test samples, resumable
%
% This is the RTL-format fixed-point reference for the final hardware path:
%   Input -> DS-Conv1/BN/ReLU -> DS-Conv2/BN/ReLU -> Pool1 ->
%   Conv/BN/ReLU -> Pool2 -> GRU -> FC1/ReLU/BN -> FC_out -> Argmax
%
% Important design rules reproduced here:
%   * signed 16-bit storage, floor quantization and saturation
%   * Pool2 and the GRU input use F12
%   * GRU gate accumulators are F30, candidate accumulators are F45
%   * GRU LUT inputs are F9; LUT outputs and hidden state are F15
%   * 256-entry Q15 sigmoid/tanh LUTs use the same address calculation as RTL
%   * hidden state is shifted and saturated after every time step
%
% No training is performed and the original mlx/model files are not changed.

    if nargin < 1 || isempty(numSamples)
        numSamples = 4;
    end
    if nargin < 3 || isempty(exportGoldenFiles)
        % Full intermediate tensors are useful for a few ModelSim samples,
        % but unnecessarily large when measuring accuracy on many samples.
        exportGoldenFiles = (numSamples <= 4);
    end

    packagePaths = EEG_Package_Paths();
    modelFile = packagePaths.model;
    datasetFile = packagePaths.dataset;
    if nargin < 2 || strlength(string(outputFolder)) == 0
        outputFolder = fullfile(packagePaths.outputs, 'fixed_inference');
    end
    outputFolder = char(outputFolder);
    weightFolder = fullfile(outputFolder, "weights");
    lutFolder = fullfile(outputFolder, "lut");
    goldenFolder = fullfile(outputFolder, "golden");
    ensureFolder(weightFolder);
    ensureFolder(lutFolder);
    ensureFolder(goldenFolder);

    assert(isfile(modelFile), "Model file not found: %s", modelFile);
    assert(isfile(datasetFile), "Dataset file not found: %s", datasetFile);
    fprintf("Loading trained model (training is not executed)...\n");
    model = load(modelFile, "net", "classNames");
    net = model.net;
    classNames = string(model.classNames(:));
    assert(numel(classNames) == 105, "Expected 105 class names.");
    requiredLayers = ["conv1_depthwise","conv1_pointwise", ...
                      "conv2_depthwise","conv2_pointwise"];
    actualLayerNames = string({net.Layers.Name});
    assert(all(ismember(requiredLayers,actualLayerNames)), ...
        "The selected model is not the DS-Conv1/DS-Conv2 architecture.");

    dataset = matfile(datasetFile);
    inputInfo = whos('-file', datasetFile, 'XTest4');
    assert(~isempty(inputInfo), "Dataset does not contain XTest4.");
    inputSize = inputInfo.size;
    totalSamples = inputSize(4);
    if isscalar(numSamples) && isinf(numSamples) && numSamples > 0
        numSamples = totalSamples;
    end
    assert(isscalar(numSamples) && numSamples >= 0 && numSamples == fix(numSamples), ...
        "numSamples must be a nonnegative integer or inf for all samples.");
    assert(numSamples <= totalSamples, ...
        "Requested %d samples, but the test set contains %d.", ...
        numSamples, totalSamples);

    F = fixedFormats();
    P = quantizeParameters(net, F);
    [sigmoidLut, tanhLut] = makeRtlLuts();

    exportParameters(P, weightFolder);
    writeMemHex(fullfile(lutFolder, "sigmoid_half_lut_q15.mem"), sigmoidLut);
    writeMemHex(fullfile(lutFolder, "tanh_half_lut_q15.mem"), tanhLut);
    save(fullfile(outputFolder, "rtl_quant_table.mat"), "F");
    writeFormatReport(fullfile(outputFolder, "rtl_formats.txt"), F);

    fprintf("Exported RTL-format weights and LUTs to:\n  %s\n", outputFolder);

    predictions = zeros(numSamples, 1, "uint8");
    winningLogits = zeros(numSamples, 1, "int16");
    if numSamples > 0
        % Load labels before inference so every completed batch can report
        % a live running accuracy instead of waiting for the entire test set.
        labelData = load(datasetFile, "YTest");
        trueSubjects = string(labelData.YTest(1:numSamples));
        [found, trueOneBased] = ismember(trueSubjects, classNames);
        assert(all(found), "Some YTest labels are absent from classNames.");
        trueClasses = uint8(trueOneBased - 1);
    else
        trueSubjects = strings(0,1);
        trueClasses = zeros(0,1,"uint8");
    end
    if exportGoldenFiles
        allActivations = initializeGoldenBuffers(numSamples);
    else
        allActivations = struct();
    end

    timer = tic;
    useParallel = ~exportGoldenFiles && numSamples >= 8 && ...
                  license('test','Distrib_Computing_Toolbox');
    if useParallel
        % Loading all 119075 samples at once needs about 1.6 GB and process
        % workers may duplicate it. Use bounded batches and checkpoint after
        % every batch so a long full-test run can resume after interruption.
        batchSize = 256;
        progressFile = fullfile(outputFolder,"full_test_progress.mat");
        completedSamples = 0;
        if isfile(progressFile)
            saved = load(progressFile,"requestedSamples", ...
                "completedSamples","predictions","winningLogits", ...
                "checkpointModelFile");
            if isfield(saved,"requestedSamples") && ...
               saved.requestedSamples == numSamples && ...
               isfield(saved,"checkpointModelFile") && ...
               strcmp(saved.checkpointModelFile,modelFile) && ...
               isfield(saved,"completedSamples") && ...
               isfield(saved,"predictions") && ...
               numel(saved.predictions) == numSamples
                completedSamples = saved.completedSamples;
                predictions = saved.predictions;
                winningLogits = saved.winningLogits;
                fprintf("Resuming saved run at sample %d/%d.\n", ...
                    completedSamples+1,numSamples);
            end
        end

        fprintf("Running %d independent samples in parallel, batch size %d...\n", ...
            numSamples,batchSize);
        sessionStart = completedSamples;
        for firstSample = completedSamples+1:batchSize:numSamples
            lastSample = min(firstSample+batchSize-1,numSamples);
            currentCount = lastSample-firstSample+1;
            inputBatch = dataset.XTest4(:,:,:,firstSample:lastSample);
            batchPredictions = zeros(currentCount,1,"uint8");
            batchLogits = zeros(currentCount,1,"int16");
            parfor localIndex = 1:currentCount
                [prediction, winningLogit] = inferOneRtlExact( ...
                    inputBatch(:,:,:,localIndex),P,F,sigmoidLut,tanhLut);
                batchPredictions(localIndex) = uint8(prediction);
                batchLogits(localIndex) = int16(winningLogit);
            end
            predictions(firstSample:lastSample) = batchPredictions;
            winningLogits(firstSample:lastSample) = batchLogits;
            completedSamples = lastSample;
            requestedSamples = numSamples;
            checkpointModelFile = modelFile;
            save(progressFile,"requestedSamples","completedSamples", ...
                "predictions","winningLogits","checkpointModelFile");

            sessionDone = completedSamples-sessionStart;
            rate = sessionDone/max(toc(timer),eps);
            remainingMinutes = (numSamples-completedSamples)/max(rate,eps)/60;
            runningCorrect = sum(predictions(1:completedSamples) == ...
                                 trueClasses(1:completedSamples));
            runningAccuracy = 100*runningCorrect/completedSamples;
            fprintf(["  %d/%d | correct=%d | accuracy=%.4f%% | " + ...
                     "%.3f samples/s | ETA %.1f hours\n"], ...
                completedSamples,numSamples,runningCorrect,runningAccuracy, ...
                rate,remainingMinutes/60);
        end
        fprintf("  Completed %d samples in this session in %.1f seconds.\n", ...
            numSamples-sessionStart,toc(timer));
    else
        for sampleIndex = 1:numSamples
            inputFloat = dataset.XTest4(:,:,:,sampleIndex);
            [prediction, winningLogit, A] = inferOneRtlExact( ...
                inputFloat, P, F, sigmoidLut, tanhLut);

            predictions(sampleIndex) = uint8(prediction);
            winningLogits(sampleIndex) = int16(winningLogit);
            if exportGoldenFiles
                allActivations = appendGolden(allActivations, A, sampleIndex);
            end

            elapsed = toc(timer);
            rate = sampleIndex / max(elapsed, eps);
            runningCorrect = sum(predictions(1:sampleIndex) == ...
                                 trueClasses(1:sampleIndex));
            runningAccuracy = 100*runningCorrect/sampleIndex;
            fprintf(["  %d/%d class=%d logit=%d correct=%d " + ...
                     "accuracy=%.4f%% %.3f samples/s\n"], ...
                sampleIndex,numSamples,prediction,winningLogit, ...
                runningCorrect,runningAccuracy,rate);
        end
    end

    if numSamples > 0
        if exportGoldenFiles
            exportGolden(allActivations, goldenFolder);
        end

        correct = predictions == trueClasses;
        accuracy = mean(correct) * 100;

        resultTable = table(uint32((0:numSamples-1)'), trueSubjects, ...
            trueClasses, predictions, winningLogits, correct, ...
            'VariableNames', {'sample_id', 'subject_label', 'true_class', ...
                              'rtl_class', 'winning_logit', 'correct'});
        writetable(resultTable, fullfile(outputFolder, "predictions.csv"));
        fprintf("RTL-format reference accuracy for selected samples: %d/%d (%.2f%%)\n", ...
            sum(correct), numSamples, accuracy);
    else
        trueClasses = zeros(0,1,"uint8");
        correct = false(0,1);
        accuracy = NaN;
    end

    report = struct();
    report.modelFile = modelFile;
    report.architecture = "DS-Conv1 + DS-Conv2";
    report.datasetFile = datasetFile;
    report.outputFolder = string(outputFolder);
    report.numSamples = numSamples;
    report.totalTestSamples = totalSamples;
    report.predictions = predictions;
    report.winningLogits = winningLogits;
    report.trueClasses = trueClasses;
    report.correct = correct;
    report.accuracyPercent = accuracy;
    report.elapsedSeconds = toc(timer);
    report.exportedGoldenFiles = exportGoldenFiles;
    report.parallelUsed = useParallel;
    save(fullfile(outputFolder, "quantize_report.mat"), "report");
end

function F = fixedFormats()
% Frozen formats used by the verified final FPGA hierarchy.
    F.input = 12;
    F.conv1Depthwise = 13; F.conv1Pointwise = 15;
    F.bn1 = 11; F.relu1 = 11;
    F.conv2Depthwise = 11; F.conv2Pointwise = 11;
    F.bn2 = 11; F.relu2 = 11; F.pool1 = 11;
    F.conv3 = 9; F.bn3 = 12; F.relu3 = 12; F.pool2 = 12;
    F.gruInput = 12; F.gruLutInput = 9; F.gruOutput = 15;
    F.flatten = 15;
    F.fc1 = 13; F.fcRelu = 13; F.fcBn = 12; F.fcOut = 10;

    F.conv1DepthwiseW = 15; F.conv1DepthwiseB = 15;
    F.conv1PointwiseW = 15; F.conv1PointwiseB = 15;
    F.conv2DepthwiseW = 15; F.conv2DepthwiseB = 15;
    F.conv2PointwiseW = 15; F.conv2PointwiseB = 15;
    F.conv3W = 15; F.conv3B = 15;
    F.bn1A = 9; F.bn1B = 11;
    F.bn2A = 14; F.bn2B = 14;
    F.bn3A = 15; F.bn3B = 14;
    F.gruWr = 15; F.gruWz = 14; F.gruWh = 15;
    F.gruUr = 15; F.gruUz = 14; F.gruUh = 15;
    F.gruBr = 15; F.gruBz = 15; F.gruBh = 15;
    F.fc1W = 15; F.fc1B = 14;
    F.fcBnA = 13; F.fcBnB = 12;
    F.fcOutW = 15; F.fcOutB = 15;
end

function P = quantizeParameters(net, F)
    layers = net.Layers;

    conv1Depthwise = findLayer(layers, "conv1_depthwise");
    conv1Pointwise = findLayer(layers, "conv1_pointwise");
    conv2Depthwise = findLayer(layers, "conv2_depthwise");
    conv2Pointwise = findLayer(layers, "conv2_pointwise");
    conv3 = findLayer(layers, "conv3");
    P.conv1DepthwiseW = quantizeInt16(conv1Depthwise.Weights, ...
        F.conv1DepthwiseW);
    P.conv1DepthwiseB = quantizeInt16(conv1Depthwise.Bias, ...
        F.conv1DepthwiseB);
    P.conv1PointwiseW = quantizeInt16(conv1Pointwise.Weights, ...
        F.conv1PointwiseW);
    P.conv1PointwiseB = quantizeInt16(conv1Pointwise.Bias, ...
        F.conv1PointwiseB);
    P.conv2DepthwiseW = quantizeInt16(conv2Depthwise.Weights, ...
        F.conv2DepthwiseW);
    P.conv2DepthwiseB = quantizeInt16(conv2Depthwise.Bias, ...
        F.conv2DepthwiseB);
    P.conv2PointwiseW = quantizeInt16(conv2Pointwise.Weights, ...
        F.conv2PointwiseW);
    P.conv2PointwiseB = quantizeInt16(conv2Pointwise.Bias, ...
        F.conv2PointwiseB);
    P.conv3W = quantizeInt16(conv3.Weights, F.conv3W);
    P.conv3B = quantizeInt16(conv3.Bias, F.conv3B);

    [P.bn1A, P.bn1B] = quantizeBatchNorm( ...
        findLayer(layers, "bn1"), F.bn1A, F.bn1B);
    [P.bn2A, P.bn2B] = quantizeBatchNorm( ...
        findLayer(layers, "bn2"), F.bn2A, F.bn2B);
    [P.bn3A, P.bn3B] = quantizeBatchNorm( ...
        findLayer(layers, "bn3"), F.bn3A, F.bn3B);

    gru = findLayer(layers, "gru");
    H = gru.NumHiddenUnits;
    W = gru.InputWeights;
    U = gru.RecurrentWeights;
    b = gru.Bias;
    if size(b,2) == 2
        b = sum(b,2);
    end
    P.gruWr = quantizeInt16(W(1:H,:), F.gruWr);
    P.gruWz = quantizeInt16(W(H+1:2*H,:), F.gruWz);
    P.gruWh = quantizeInt16(W(2*H+1:3*H,:), F.gruWh);
    P.gruUr = quantizeInt16(U(1:H,:), F.gruUr);
    P.gruUz = quantizeInt16(U(H+1:2*H,:), F.gruUz);
    P.gruUh = quantizeInt16(U(2*H+1:3*H,:), F.gruUh);
    P.gruBr = quantizeInt16(b(1:H), F.gruBr);
    P.gruBz = quantizeInt16(b(H+1:2*H), F.gruBz);
    P.gruBh = quantizeInt16(b(2*H+1:3*H), F.gruBh);

    fc1 = findLayer(layers, "fc_1");
    P.fc1W = quantizeInt16(fc1.Weights, F.fc1W);
    P.fc1B = quantizeInt16(fc1.Bias, F.fc1B);
    [P.fcBnA, P.fcBnB] = quantizeBatchNorm( ...
        findLayer(layers, "bn_2"), F.fcBnA, F.fcBnB);
    fcOut = findLayer(layers, "fc_out");
    P.fcOutW = quantizeInt16(fcOut.Weights, F.fcOutW);
    P.fcOutB = quantizeInt16(fcOut.Bias, F.fcOutB);
end

function layer = findLayer(layers, name)
    index = find(strcmp({layers.Name}, name), 1);
    assert(~isempty(index), "Cannot find layer: %s", name);
    layer = layers(index);
end

function [Aq, Bq] = quantizeBatchNorm(layer, Fa, Fb)
    A = layer.Scale ./ sqrt(layer.TrainedVariance + layer.Epsilon);
    B = layer.Offset - layer.TrainedMean .* layer.Scale ./ ...
        sqrt(layer.TrainedVariance + layer.Epsilon);
    Aq = quantizeInt16(A, Fa);
    Bq = quantizeInt16(B, Fb);
end

function [prediction, winningLogit, A] = inferOneRtlExact(X, P, F, sigLut, tanhLut)
    A.input = quantizeInt16(X, F.input);

    A.conv1Depthwise = depthwiseConvRtl(A.input, ...
        P.conv1DepthwiseW, P.conv1DepthwiseB, 12, 14);
    A.conv1Pointwise = convRtl(A.conv1Depthwise, ...
        P.conv1PointwiseW, P.conv1PointwiseB, 13, 13);
    A.bn1 = bnRtl(A.conv1Pointwise, P.bn1A, P.bn1B, 13, 13);
    A.relu1 = reluRtl(A.bn1, 0);

    A.conv2Depthwise = depthwiseConvRtl(A.relu1, ...
        P.conv2DepthwiseW, P.conv2DepthwiseB, 11, 15);
    A.conv2Pointwise = convRtl(A.conv2Depthwise, ...
        P.conv2PointwiseW, P.conv2PointwiseB, 11, 15);
    A.bn2 = bnRtl(A.conv2Pointwise, P.bn2A, P.bn2B, 11, 14);
    A.relu2 = reluRtl(A.bn2, 0);
    A.pool1 = poolRtl(A.relu2, 1, 10, 1, 8, 0);

    A.conv3 = convRtl(A.pool1, P.conv3W, P.conv3B, 11, 17);
    A.bn3 = bnRtl(A.conv3, P.bn3A, P.bn3B, 10, 12);
    A.relu3 = reluRtl(A.bn3, 0);
    A.pool2 = poolRtl(A.relu3, 1, 10, 1, 8, 0);

    A.gru = gruRtl(A.pool2, P, sigLut, tanhLut);
    A.flatten = reshape(A.gru, [], 1);

    A.fc1 = fcRtl(A.flatten, P.fc1W, P.fc1B, 16, 17);
    A.fcRelu = reluRtl(A.fc1, 0);
    A.fcBn = bnRtl(reshape(A.fcRelu,1,1,[]), ...
        P.fcBnA, P.fcBnB, 14, 14);
    A.fcBn = reshape(A.fcBn, [], 1);
    A.fcOut = fcRtl(A.fcBn, P.fcOutW, P.fcOutB, 12, 17);

    logits = double(A.fcOut(:));
    [winningLogit, oneBased] = max(logits);
    prediction = oneBased - 1;
    winningLogit = int16(winningLogit);
end

function Y = depthwiseConvRtl(X, W, B, biasShift, outputShift)
    Xd = double(X);
    Wd = double(W);
    Bd = double(B(:));
    [inH,inW,inCh] = size3(Xd);
    kH = size(Wd,1);
    kW = size(Wd,2);
    assert(numel(Bd) == inCh, "Depthwise bias/channel mismatch.");
    assert(size(Wd,5) == inCh || inCh == 1, ...
        "Depthwise weight/channel mismatch.");

    outH = inH-kH+1;
    outW = inW-kW+1;
    positions = outH*outW;
    Yd = zeros(outH,outW,inCh);
    for ch = 1:inCh
        patches = zeros(kH*kW,positions);
        position = 0;
        for ow = 1:outW
            for oh = 1:outH
                position = position+1;
                patch = Xd(oh:oh+kH-1,ow:ow+kW-1,ch);
                patches(:,position) = patch(:);
            end
        end
        if inCh == 1
            weights = reshape(Wd,kH*kW,1);
        else
            weights = reshape(Wd(:,:,1,1,ch),kH*kW,1);
        end
        accumulator = weights.'*patches;
        raw = floor((accumulator + Bd(ch)*2^biasShift) / ...
                    2^outputShift);
        Yd(:,:,ch) = reshape(sat16(raw.'),outH,outW);
    end
    Y = int16(Yd);
end

function Y = convRtl(X, W, B, biasShift, outputShift)
    Xd = double(X);
    Wd = double(W);
    Bd = double(B(:));
    [inH,inW,inCh] = size3(Xd);
    [kH,kW,weightInCh,outCh] = size4(Wd);
    assert(inCh == weightInCh, "Convolution channel mismatch.");
    outH = inH-kH+1;
    outW = inW-kW+1;
    % Build the same column-major kernel vectors issued by conv_controller:
    % kh first, then kw, then input channel. All operands are integer codes;
    % their exact sums remain below double's 53-bit integer precision.
    kernelSize = kH*kW*inCh;
    positions = outH*outW;
    patches = zeros(kernelSize,positions);
    position = 0;
    for ow = 1:outW
        for oh = 1:outH
            position = position+1;
            patch = Xd(oh:oh+kH-1,ow:ow+kW-1,:);
            patches(:,position) = patch(:);
        end
    end
    weights = reshape(Wd,kernelSize,outCh);
    accumulators = weights.'*patches;
    raw = floor((accumulators + Bd*2^biasShift) / 2^outputShift);
    Yd = reshape(sat16(raw.'),outH,outW,outCh);
    Y = int16(Yd);
end

function Y = bnRtl(X, A, B, biasShift, outputShift)
    Xd = double(X);
    Ad = double(A(:));
    Bd = double(B(:));
    [h,w,c] = size3(Xd);
    assert(numel(Ad) == c && numel(Bd) == c, "BN channel mismatch.");
    Yd = zeros(h,w,c);
    for ch = 1:c
        raw = floor((Xd(:,:,ch)*Ad(ch) + Bd(ch)*2^biasShift) / ...
                    2^outputShift);
        Yd(:,:,ch) = sat16(raw);
    end
    Y = int16(Yd);
end

function Y = reluRtl(X, leftShift)
    raw = double(X);
    raw(raw < 0) = 0;
    raw = raw * 2^leftShift;
    Y = int16(sat16(raw));
end

function Y = poolRtl(X, poolH, poolW, strideH, strideW, leftShift)
    Xd = double(X);
    [inH,inW,inCh] = size3(Xd);
    outH = floor((inH-poolH)/strideH)+1;
    outW = floor((inW-poolW)/strideW)+1;
    Yd = zeros(outH,outW,inCh);
    for ch = 1:inCh
        for ow = 1:outW
            for oh = 1:outH
                window = Xd((oh-1)*strideH+(1:poolH), ...
                            (ow-1)*strideW+(1:poolW), ch);
                Yd(oh,ow,ch) = sat16(max(window,[],'all') * 2^leftShift);
            end
        end
    end
    Y = int16(Yd);
end

function Y = gruRtl(pool2, P, sigLut, tanhLut)
    % pool2 is [18,1,15]. The RTL reads time + 18*feature.
    X = reshape(double(pool2), size(pool2,1), size(pool2,3)).'; % [15,18]
    H = size(P.gruWz,1);
    T = size(X,2);
    hidden = zeros(H,1);
    Yd = zeros(H,T);

    Wr=double(P.gruWr); Wz=double(P.gruWz); Wh=double(P.gruWh);
    Ur=double(P.gruUr); Uz=double(P.gruUz); Uh=double(P.gruUh);
    br=double(P.gruBr(:)); bz=double(P.gruBz(:)); bh=double(P.gruBh(:));

    for t = 1:T
        xt = X(:,t);
        resetGate = zeros(H,1);
        updateGate = zeros(H,1);

        for n = 1:H
            resetAcc = br(n)*2^15 + sum((Wr(n,:).'.*xt)*2^3) + ...
                       sum(Ur(n,:).'.*hidden);
            updateAcc = bz(n)*2^15 + sum((Wz(n,:).'.*xt)*2^4) + ...
                        sum((Uz(n,:).'.*hidden)*2^1);
            resetInput = sat16(floor(resetAcc/2^(30-9)));
            updateInput = sat16(floor(updateAcc/2^(30-9)));
            resetGate(n) = sigmoidRtl(resetInput, sigLut);
            updateGate(n) = sigmoidRtl(updateInput, sigLut);
        end

        hiddenNext = zeros(H,1);
        for n = 1:H
            gatedHidden = resetGate .* hidden; % F15*F15 = F30
            candidateAcc = bh(n)*2^30 + sum((Wh(n,:).'.*xt)*2^18) + ...
                           sum(Uh(n,:).'.*gatedHidden);
            candidateInput = sat16(floor(candidateAcc/2^(45-9)));
            candidate = tanhRtl(candidateInput, tanhLut);
            mixed = (32768-updateGate(n))*candidate + ...
                    updateGate(n)*hidden(n);
            hiddenNext(n) = sat16(floor(mixed/2^15));
        end
        hidden = hiddenNext;
        Yd(:,t) = hidden;
    end
    Y = int16(Yd);
end

function value = sigmoidRtl(inputCode, lut)
    raw = double(inputCode);
    magnitude = min(abs(raw),3193);
    if magnitude >= 3193
        address = 255;
    else
        address = floor(magnitude*83741/2^20);
    end
    rom = double(lut(address+1));
    if raw < 0
        value = 32768-rom;
    else
        value = rom;
    end
end

function value = tanhRtl(inputCode, lut)
    raw = double(inputCode);
    magnitude = min(abs(raw),1774);
    if magnitude >= 1774
        address = 255;
    else
        address = floor(magnitude*150725/2^20);
    end
    value = double(lut(address+1));
    if raw < 0
        value = -value;
    end
end

function Y = fcRtl(X, W, B, biasShift, outputShift)
    Xd = double(X(:));
    Wd = double(W);
    Bd = double(B(:));
    assert(size(Wd,2) == numel(Xd), "FC input size mismatch.");
    raw = floor((Wd*Xd + Bd*2^biasShift) / 2^outputShift);
    Y = int16(sat16(raw));
end

function [sigmoidLut, tanhLut] = makeRtlLuts()
    index = (0:255)';
    sigmoidX = index * ((3193/512)/255);
    tanhX = index * ((1774/512)/255);
    sigmoidLut = int16(min(32767, round((1./(1+exp(-sigmoidX)))*32768)));
    tanhLut = int16(min(32767, round(tanh(tanhX)*32768)));
end

function Y = quantizeInt16(X, fractionalBits)
    Y = int16(sat16(floor(double(X)*2^fractionalBits)));
end

function Y = sat16(X)
    Y = min(max(X,-32768),32767);
end

function [a,b,c] = size3(X)
    s = size(X); s(end+1:3) = 1;
    a=s(1); b=s(2); c=s(3);
end

function [a,b,c,d] = size4(X)
    s = size(X); s(end+1:4) = 1;
    a=s(1); b=s(2); c=s(3); d=s(4);
end

function exportParameters(P, folder)
    names = fieldnames(P);
    fileNames = struct( ...
        'conv1DepthwiseW','conv1_depthwise_W.mem', ...
        'conv1DepthwiseB','conv1_depthwise_b.mem', ...
        'conv1PointwiseW','conv1_pointwise_W.mem', ...
        'conv1PointwiseB','conv1_pointwise_b.mem', ...
        'conv2DepthwiseW','conv2_depthwise_W.mem', ...
        'conv2DepthwiseB','conv2_depthwise_b.mem', ...
        'conv2PointwiseW','conv2_pointwise_W.mem', ...
        'conv2PointwiseB','conv2_pointwise_b.mem', ...
        'conv3W','conv3_W.mem','conv3B','conv3_b.mem', ...
        'bn1A','bn1_A.mem','bn1B','bn1_B.mem', ...
        'bn2A','bn2_A.mem','bn2B','bn2_B.mem', ...
        'bn3A','bn3_A.mem','bn3B','bn3_B.mem', ...
        'gruWr','gru_Wr.mem','gruWz','gru_Wz.mem','gruWh','gru_Wh.mem', ...
        'gruUr','gru_Ur.mem','gruUz','gru_Uz.mem','gruUh','gru_Uh.mem', ...
        'gruBr','gru_br.mem','gruBz','gru_bz.mem','gruBh','gru_bh.mem', ...
        'fc1W','fc_1_W.mem','fc1B','fc_1_b.mem', ...
        'fcBnA','bn_2_A.mem','fcBnB','bn_2_B.mem', ...
        'fcOutW','fc_out_W.mem','fcOutB','fc_out_b.mem');
    for k = 1:numel(names)
        name = names{k};
        writeMemHex(fullfile(folder,fileNames.(name)),P.(name));
    end
end

function buffers = initializeGoldenBuffers(n)
    buffers.input=zeros(3360,n,'int16');
    buffers.conv1Depthwise=zeros(3120,n,'int16');
    buffers.conv1Pointwise=zeros(65520,n,'int16');
    buffers.bn1=zeros(65520,n,'int16'); buffers.relu1=zeros(65520,n,'int16');
    buffers.conv2Depthwise=zeros(60648,n,'int16');
    buffers.conv2Pointwise=zeros(57760,n,'int16');
    buffers.bn2=zeros(57760,n,'int16');
    buffers.relu2=zeros(57760,n,'int16'); buffers.pool1=zeros(6840,n,'int16');
    buffers.conv3=zeros(3780,n,'int16'); buffers.bn3=zeros(3780,n,'int16');
    buffers.relu3=zeros(3780,n,'int16'); buffers.pool2=zeros(270,n,'int16');
    buffers.flatten=zeros(162,n,'int16'); buffers.fc1=zeros(40,n,'int16');
    buffers.fcRelu=zeros(40,n,'int16'); buffers.fcBn=zeros(40,n,'int16');
    buffers.fcOut=zeros(105,n,'int16');
end

function buffers = appendGolden(buffers,A,index)
    names=fieldnames(buffers);
    for k=1:numel(names)
        name=names{k};
        buffers.(name)(:,index)=A.(name)(:);
    end
end

function exportGolden(B,folder)
    map = struct('input','q_in_act.mem', ...
        'conv1Depthwise','q_conv1_depthwise_act.mem', ...
        'conv1Pointwise','q_conv1_pointwise_act.mem', ...
        'bn1','q_bn1_act.mem','relu1','q_relu1_act.mem', ...
        'conv2Depthwise','q_conv2_depthwise_act.mem', ...
        'conv2Pointwise','q_conv2_pointwise_act.mem', ...
        'bn2','q_bn2_act.mem', ...
        'relu2','q_relu2_act.mem','pool1','q_pool1_act.mem', ...
        'conv3','q_conv3_act.mem','bn3','q_bn3_act.mem', ...
        'relu3','q_relu3_act.mem','pool2','q_pool2_act.mem', ...
        'flatten','q_flatten_act.mem','fc1','q_fc_1_act.mem', ...
        'fcRelu','q_fc_relu_act.mem','fcBn','q_bn_2_act.mem', ...
        'fcOut','q_fc_out_act.mem');
    names=fieldnames(B);
    for k=1:numel(names)
        name=names{k}; writeMemHex(fullfile(folder,map.(name)),B.(name));
    end
end

function writeMemHex(filename,X)
    fid=fopen(filename,'w');
    assert(fid~=-1,"Cannot open output file: %s",filename);
    cleanup=onCleanup(@()fclose(fid));
    values=typecast(int16(X(:)),'uint16');
    fprintf(fid,'%04X\n',values);
    clear cleanup;
end

function writeFormatReport(filename,F)
    fid=fopen(filename,'w'); assert(fid~=-1,"Cannot open: %s",filename);
    cleanup=onCleanup(@()fclose(fid));
    fprintf(fid,"All values use signed 16-bit storage. F means fractional bits.\n\n");
    fprintf(fid,"Input F%d\n",F.input);
    fprintf(fid,"DSConv1 DW F%d -> PW F%d -> BN1 F%d -> ReLU1 F%d\n", ...
        F.conv1Depthwise,F.conv1Pointwise,F.bn1,F.relu1);
    fprintf(fid,"DSConv2 DW F%d -> PW F%d -> BN2 F%d -> ReLU2 F%d -> Pool1 F%d\n", ...
        F.conv2Depthwise,F.conv2Pointwise,F.bn2,F.relu2,F.pool1);
    fprintf(fid,"Conv3 F%d -> BN3 F%d -> ReLU3 F%d -> Pool2 F%d\n",F.conv3,F.bn3,F.relu3,F.pool2);
    fprintf(fid,"GRU input F%d -> LUT input F%d -> hidden/output F%d\n",F.gruInput,F.gruLutInput,F.gruOutput);
    fprintf(fid,"Flatten F%d -> FC1 F%d -> ReLU F%d -> BN F%d -> FC out F%d\n",F.flatten,F.fc1,F.fcRelu,F.fcBn,F.fcOut);
    clear cleanup;
end

function ensureFolder(folder)
    if ~isfolder(folder), mkdir(folder); end
end
