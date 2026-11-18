function p = EEG_Package_Paths()
% Resolve all package files relative to this file, independently of cwd.
    p.root = fileparts(fileparts(mfilename('fullpath')));
    p.code = fullfile(p.root,'01_MATLAB_Code');
    p.model = fullfile(p.root,'02_Final_Trained_Model', ...
        'EEG_CNN_GRU_dsconv1_dsconv2_mixedSess.mat');
    p.dataset = fullfile(p.root,'03_PreSplit_Dataset', ...
        'EEG_105_stride10_mixedSess.mat');
    p.reference = fullfile(p.root,'04_Reference_Quantization_Results');
    p.outputs = fullfile(p.root,'05_New_Simulation_Results');
    p.validation = fullfile(p.root,'07_Validation_Records');
    assert(isfile(p.model),'Trained model missing: %s',p.model);
    assert(isfile(p.dataset),'Pre-split dataset missing: %s',p.dataset);
end
