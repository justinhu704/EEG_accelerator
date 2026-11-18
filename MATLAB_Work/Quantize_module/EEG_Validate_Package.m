function verification = EEG_Validate_Package(numSamples,outputRoot)
% Check split metadata/labels and rerun a few float/fixed inference samples.
% Does not load complete input tensors, train, split data or read raw EDF.
    if nargin < 1, numSamples = 4; end
    assert(isscalar(numSamples) && isfinite(numSamples) && ...
        numSamples >= 1 && numSamples == fix(numSamples),'Use a positive sample count.');
    p = EEG_Package_Paths();
    if nargin < 2 || strlength(string(outputRoot))==0
        stamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
        outputRoot = fullfile(p.outputs,['quick_check_' stamp]);
    end
    assert(~isfolder(outputRoot),'Use a new output folder: %s',outputRoot);
    mkdir(outputRoot);
    verification = struct('status','RUNNING','matlabVersion',version, ...
        'packageRoot',p.root,'checkedAt',char(datetime('now')), ...
        'smokeSamples',numSamples,'fullAccuracyRerun',false);
    verification.toolboxes = ver;
    info = whos('-file',p.dataset);
    m = load(p.model,'classNames','acc','inputSize','mu','sig');
    classNames = string(m.classNames(:));
    assert(numel(classNames)==105 && numel(unique(classNames))==105, ...
        'Expected 105 unique classes.');
    assert(isequal(m.inputSize,[21 160]),'Unexpected model input size.');
    assert(isequal(size(m.mu),[21 1]) && isequal(size(m.sig),[21 1]), ...
        'Saved normalization metadata is missing.');
    verification.savedFloatAccuracyPercent = 100*m.acc;
    verification.classCount = numel(classNames);
    pairs = {'train','XTr4','YTr';'validation','XVal4','YVal';'test','XTest4','YTest'};
    for k = 1:size(pairs,1)
        x = info(strcmp({info.name},pairs{k,2}));
        y = info(strcmp({info.name},pairs{k,3}));
        assert(numel(x)==1 && numel(y)==1,'Missing split variables.');
        dims = x.size;
        assert(numel(dims)==4 && isequal(dims(1:3),[21 160 1]), ...
            'Unexpected %s input shape.',pairs{k,1});
        labelData = load(p.dataset,pairs{k,3});
        labels = string(labelData.(pairs{k,3})(:));
        assert(dims(4)==numel(labels),'%s input/label count mismatch.',pairs{k,1});
        assert(all(ismember(labels,classNames)),'Unknown split labels.');
        assert(numel(unique(labels))==105,'A split does not include all classes.');
        verification.splits.(pairs{k,1}) = struct('inputVariable',pairs{k,2}, ...
            'labelVariable',pairs{k,3},'shape',dims, ...
            'sampleCount',numel(labels),'classCount',numel(unique(labels)));
        fprintf('%s: %s / %s, %d samples, 105 classes\n', ...
            pairs{k,1},pairs{k,2},pairs{k,3},numel(labels));
    end
    verification.float = EEG_Float_Evaluate(numSamples,fullfile(outputRoot,'float'));
    fixed = EEG_Quantize(numSamples,fullfile(outputRoot,'fixed'),true);
    ref = readtable(fullfile(p.reference,'predictions.csv'),'TextType','string');
    assert(height(ref)==verification.splits.test.sampleCount, ...
        'Historical full predictions are incomplete.');
    assert(isequal(double(ref.sample_id),(0:height(ref)-1)'), ...
        'Historical sample IDs are not sequential.');
    labels = load(p.dataset,'YTest');
    [found,referenceTruth] = ismember(string(labels.YTest(:)),classNames);
    assert(all(found) && isequal(double(ref.true_class),double(referenceTruth-1)), ...
        'Historical truth labels differ from the current cache/model order.');
    rows = 1:numSamples;
    assert(isequal(double(fixed.predictions(:)),double(ref.rtl_class(rows))), ...
        'Fixed predictions differ from the historical run.');
    assert(isequal(double(fixed.winningLogits(:)),double(ref.winning_logit(rows))), ...
        'Fixed logits differ from the historical run.');
    assert(isequal(double(fixed.trueClasses(:)),double(ref.true_class(rows))), ...
        'Fixed truth labels differ from the historical run.');
    nCompared = 0;
    for folder = {'weights','lut'}
        oldFiles = dir(fullfile(p.reference,folder{1},'*.mem'));
        newFiles = dir(fullfile(outputRoot,'fixed',folder{1},'*.mem'));
        assert(~isempty(oldFiles) && numel(newFiles)==numel(oldFiles), ...
            'Exported parameter file count mismatch.');
        for k = 1:numel(oldFiles)
            old = readBytes(fullfile(oldFiles(k).folder,oldFiles(k).name));
            new = readBytes(fullfile(outputRoot,'fixed',folder{1},oldFiles(k).name));
            assert(isequal(old,new),'Parameter bytes differ: %s',oldFiles(k).name);
            nCompared = nCompared + 1;
        end
    end
    verification.fixed = rmfield(fixed,{'predictions','winningLogits','trueClasses','correct'});
    verification.fixed.correctCount = sum(fixed.correct);
    verification.historicalFixedCorrect = sum(ref.correct);
    verification.historicalFixedAccuracyPercent = 100*mean(ref.correct);
    verification.fixedMatchesReferenceForSmokeSamples = true;
    verification.matchingExportedParameterFiles = nCompared;
    verification.status = 'PASS';
    save(fullfile(outputRoot,'verification.mat'),'verification');
    fid = fopen(fullfile(outputRoot,'verification.json'),'w','n','UTF-8');
    assert(fid>=0,'Cannot write verification report.');
    cleanup = onCleanup(@() fclose(fid));
    fwrite(fid,jsonencode(verification,'PrettyPrint',true),'char');
    fprintf('PASS: %d float/fixed samples; %d parameter/LUT files match.\n',numSamples,nCompared);
    fprintf('Report: %s\n',fullfile(outputRoot,'verification.json'));
end

function b = readBytes(path)
    fid = fopen(path,'rb');
    assert(fid>=0,'Missing parameter file: %s',path);
    cleanup = onCleanup(@() fclose(fid));
    b = fread(fid,inf,'*uint8');
end
