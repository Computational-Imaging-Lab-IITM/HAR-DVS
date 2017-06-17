%% clear workspace %%
clear;
close;
clc;


%% get the current folder and cd to it %%
currentFolder = regexprep(mfilename('fullpath'), mfilename(), '');
cd(currentFolder);


%% dataset we want to extract features for and properties dependent on it %%
datasetName = 'IITM_DVS_10';

% number of folds %
if strcmp(datasetName, 'UCF11_DVS') == 1
    % UCF11 dataset %
    num_folds = 25;
    num_classes = 11;
elseif strcmp(datasetName, 'IITM_DVS_10') == 1
    % IITM_DVS_10 dataset %
    num_folds = 12;
    num_classes = 10;
elseif strcmp(datasetName, 'IITM_DVS_12') == 1
    % IITM_DVS_12 dataset %
    num_folds = 6;
    num_classes = 12;
else
    % unknown dataset %
    num_folds = 0;
    num_classes = 0;
end

% path to the dataset %
dataset_path = fullfile(currentFolder, '/data/', datasetName, '/original_data/');
% path to the features extracted %
features_path = fullfile(currentFolder, '/data/', datasetName, '/features_extracted/');
% path to the encoded data folder %
encoded_data_path = fullfile(currentFolder, '/data/', datasetName, '/encoded_data/');


%% running the k-fold %%
tic
fprintf('******************************************************\n');
fprintf('***** Running %d-fold on the %s dataset *****\n', num_folds, datasetName);
fprintf('******************************************************\n\n');

all_accuracy_xy_xt_yt = zeros(num_folds, 1);
all_accuracy_xy_xt_yt_MBHx_MBHy = zeros(num_folds, 1);
all_accuracy_xy_xt_yt_HoF = zeros(num_folds, 1);
all_accuracy_HoG = zeros(num_folds, 1);
all_accuracy_HoF = zeros(num_folds, 1);
all_accuracy_MBHx_MBHy = zeros(num_folds, 1);
all_accuracy_xy = zeros(num_folds, 1);
all_accuracy_xt = zeros(num_folds, 1);
all_accuracy_yt = zeros(num_folds, 1);

summed_confusion_xy_xt_yt = zeros(num_classes, num_classes);
summed_confusion_xy_xt_yt_MBHx_MBHy = zeros(num_classes, num_classes);
summed_confusion_xy_xt_yt_HoF = zeros(num_classes, num_classes);
summed_confusion_HoG = zeros(num_classes, num_classes);
summed_confusion_HoF = zeros(num_classes, num_classes);
summed_confusion_MBHx_MBHy = zeros(num_classes, num_classes);
summed_confusion_xy = zeros(num_classes, num_classes);
summed_confusion_xt = zeros(num_classes, num_classes);
summed_confusion_yt = zeros(num_classes, num_classes);

for K=1:num_folds
    fprintf('Starting the fold, K = %d\n', K);
    % generate codebooks for dense-features and motion-maps and encode all videos %
    generate_codebook(K, datasetName, features_path, encoded_data_path, currentFolder);
        
    % build SVM and combine the accuracies %
    [accuracy_xy_xt_yt, accuracy_xy_xt_yt_MBHx_MBHy, accuracy_xy_xt_yt_HoF, accuracy_HoG, accuracy_HoF, accuracy_MBHx_MBHy, accuracy_xy, accuracy_xt, accuracy_yt, confusion_xy_xt_yt, confusion_xy_xt_yt_MBHx_MBHy, confusion_xy_xt_yt_HoF, confusion_HoG, confusion_HoF, confusion_MBHx_MBHy, confusion_xy, confusion_xt, confusion_yt] = svm_loo(encoded_data_path, num_classes);
    
    % updating accuracies %
    all_accuracy_xy_xt_yt(K) = accuracy_xy_xt_yt;
    all_accuracy_xy_xt_yt_MBHx_MBHy(K) = accuracy_xy_xt_yt_MBHx_MBHy;
    all_accuracy_xy_xt_yt_HoF(K) = accuracy_xy_xt_yt_HoF;
    all_accuracy_HoG(K) = accuracy_HoG;
    all_accuracy_HoF(K) = accuracy_HoF;
    all_accuracy_MBHx_MBHy(K) = accuracy_MBHx_MBHy;
    all_accuracy_xy(K) = accuracy_xy;
    all_accuracy_xt(K) = accuracy_xt;
    all_accuracy_yt(K) = accuracy_yt;
    
    % updating the confusion matrix %
    summed_confusion_xy_xt_yt = summed_confusion_xy_xt_yt + confusion_xy_xt_yt;
    summed_confusion_xy_xt_yt_MBHx_MBHy = summed_confusion_xy_xt_yt_MBHx_MBHy + confusion_xy_xt_yt_MBHx_MBHy;
    summed_confusion_xy_xt_yt_HoF = summed_confusion_xy_xt_yt_HoF + confusion_xy_xt_yt_HoF;
    summed_confusion_HoG = summed_confusion_HoG + confusion_HoG;
    summed_confusion_HoF = summed_confusion_HoF + confusion_HoF;
    summed_confusion_MBHx_MBHy = summed_confusion_MBHx_MBHy + confusion_MBHx_MBHy;
    summed_confusion_xy = summed_confusion_xy + confusion_xy;
    summed_confusion_xt = summed_confusion_xt + confusion_xt;
    summed_confusion_yt = summed_confusion_yt + confusion_yt;
    
    % storing the accuracies %
    cd(currentFolder);
    accuracies_folder = strcat('./workspace_data/accuracies/', datasetName, '/');
    save(strcat(accuracies_folder, 'all_accuracy_xy_xt_yt'), 'all_accuracy_xy_xt_yt', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_xy_xt_yt_MBHx_MBHy'), 'all_accuracy_xy_xt_yt_MBHx_MBHy', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_xy_xt_yt_HoF'), 'all_accuracy_xy_xt_yt_HoF', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_HoG'), 'all_accuracy_HoG', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_HoF'), 'all_accuracy_HoF', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_MBHx_MBHy'), 'all_accuracy_MBHx_MBHy', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_xy'), 'all_accuracy_xy', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_xt'), 'all_accuracy_xt', '-v7.3');
    save(strcat(accuracies_folder, 'all_accuracy_yt'), 'all_accuracy_yt', '-v7.3');
    
    % saving the confusion matrices %
    cd(currentFolder);
    confusion_folder = strcat('./workspace_data/confusion_matrices/', datasetName, '/');
    save(strcat(confusion_folder, 'summed_confusion_xy_xt_yt'), 'summed_confusion_xy_xt_yt', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_xy_xt_yt_MBHx_MBHy'), 'summed_confusion_xy_xt_yt_MBHx_MBHy', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_xy_xt_yt_HoF'), 'summed_confusion_xy_xt_yt_HoF', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_HoG'), 'summed_confusion_HoG', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_HoF'), 'summed_confusion_HoF', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_MBHx_MBHy'), 'summed_confusion_MBHx_MBHy', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_xy'), 'summed_confusion_xy', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_xt'), 'summed_confusion_xt', '-v7.3');
    save(strcat(confusion_folder, 'summed_confusion_yt'), 'summed_confusion_yt', '-v7.3');
end
toc


%% printing the accuracies %%


