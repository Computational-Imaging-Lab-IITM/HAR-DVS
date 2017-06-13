%% clear workspace %%
clear;
close;
clc;


%% get the current folder and cd to it %%
currentFolder = regexprep(mfilename('fullpath'), mfilename(), '');
cd(currentFolder);


%% dataset we want to extract features for and properties dependent on it %%
datasetName = 'UCF11';

% number of folds %
if strcmp(datasetName, 'UCF11') == 1
    % UCF11 dataset %
    num_folds = 25;
elseif strcmp(datasetName, 'IITM_DVS_10') == 1
    % IITM_DVS_10 dataset %
    num_folds = 12;
elseif strcmp(datasetName, 'IITM_DVS_12') == 1
    % IITM_DVS_12 dataset %
    num_folds = 6;
else
    % unknown dataset %
    num_folds = 0;
end

% path to the dataset %
dataset_path = fullfile(currentFolder, '/data/', datasetName, '/original_data/');
% path to the features extracted %
features_path = fullfile(currentFolder, '/data/', datasetName, '/features_extracted/');
% path to the encoded data folder %
encoded_data_path = fullfile(currentFolder, '/data/', datasetName, '/encoded_data/');

% calculate the number of classes in the dataset %
num_classes = length(dir(dataset_path));


%% running the k-fold %%
tic
fprintf('******************************************************\n');
fprintf('***** Running %d-fold on the %s dataset *****\n', num_folds, datasetName);
fprintf('******************************************************\n\n');

accuracies
for K=1:num_folds
    % generate dense features codebook and encode all videos %
    generate_codebook(K, datasetName, features_path, encoded_data_path, currentFolder);
    
    % encode motion maps of all videos %
    
    
    % build SVM and combine the accuracies %
    [accuracies, conf_mats] = svm_loo(encoded_data_path, num_classes);
    
end
toc


%% printing the accuracies %%



