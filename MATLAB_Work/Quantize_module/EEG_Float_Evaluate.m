function report = EEG_Float_Evaluate(numSamples,outputFolder)
% Evaluate the saved floating-point network; never train or read raw EDF.
% Inputs in XTest4 are already preprocessed. Do not normalize them again.
    if nargin < 1, numSamples = 4; end
    p = EEG_Package_Paths();
    if nargin < 2 || strlength(string(outputFolder)) == 0
        outputFolder = fullfile(p.outputs,'float_inference');
    end
    if ~isfolder(outputFolder), mkdir(outputFolder); end
    m = load(p.model,'net','classNames');
    classNames = string(m.classNames(:));
    info = whos('-file',p.dataset,'XTest4');
    assert(~isempty(info),'XTest4 is missing.');
    dims = info.size;
    assert(numel(dims)==4 && isequal(dims(1:3),[21 160 1]), ...
        'Unexpected input shape.');
    total = dims(4);
    if isequal(numSamples,inf), numSamples = total; end
    assert(isscalar(numSamples) && isfinite(numSamples) && ...
        numSamples >= 1 && numSamples == fix(numSamples) && numSamples <= total, ...
        'numSamples must be 1..%d or inf.',total);
    labels = load(p.dataset,'YTest');
    truth = string(labels.YTest(:));
    assert(numel(truth)==total,'Test input/label counts differ.');
    [found,trueIndex] = ismember(truth(1:numSamples),classNames);
    assert(all(found),'Unknown test labels.');
    assert(numel(classNames)==105,'Expected 105 classes.');
    netClasses = string(m.net.Layers(end).Classes(:));
    [found,modelToNet] = ismember(classNames,netClasses);
    assert(all(found) && numel(netClasses)==numel(classNames), ...
        'Network classes differ from saved classNames.');
    dataset = matfile(p.dataset);
    predictedSubjects = strings(numSamples,1);
    scores = zeros(numSamples,numel(classNames),'single');
    timer = tic;
    for first = 1:256:numSamples
        last = min(first+255,numSamples);
        batch = dataset.XTest4(:,:,:,first:last);
        [pred,batchScores] = classify(m.net,batch, ...
            'ExecutionEnvironment','cpu','MiniBatchSize',64);
        predictedSubjects(first:last) = string(pred(:));
        scores(first:last,:) = single(batchScores(:,modelToNet));
        fprintf('Float inference %d/%d (%.1f seconds)\n',last,numSamples,toc(timer));
    end
    [found,predictedIndex] = ismember(predictedSubjects,classNames);
    assert(all(found),'Network returned an unknown label.');
    correct = predictedSubjects == truth(1:numSamples);
    T = table(uint32((0:numSamples-1)'),truth(1:numSamples), ...
        uint8(trueIndex-1),predictedSubjects,uint8(predictedIndex-1),correct, ...
        'VariableNames',{'sample_id','subject_label','true_class', ...
        'predicted_subject','predicted_class','correct'});
    writetable(T,fullfile(outputFolder,'predictions.csv'));
    report = struct('modelFile',p.model,'datasetFile',p.dataset, ...
        'outputFolder',outputFolder,'numSamples',numSamples, ...
        'totalTestSamples',total,'correctCount',sum(correct), ...
        'accuracyPercent',100*mean(correct),'elapsedSeconds',toc(timer));
    save(fullfile(outputFolder,'float_report.mat'), ...
        'report','predictedSubjects','scores','classNames');
    fprintf('Float accuracy: %d/%d (%.4f%%)\n',sum(correct),numSamples,report.accuracyPercent);
end
