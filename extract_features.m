%% clear workspace %%
clear;
close;
clc;


%% get the current folder and cd to it %%
currentFolder = regexprep(mfilename('fullpath'), mfilename(), '');
cd(currentFolder);


%% dataset we want to extract features for %%
datasetName = 'IITM_DVS_10';
% path to the dataset %
dataset_path = fullfile(currentFolder, '/data/', datasetName, '/original_data/');
data_dir = dir(dataset_path);
% path to the features extracted %
features_path = fullfile(currentFolder, '/data/', datasetName, '/features_extracted/');
features_dir = dir(features_path);

% remove hidden folders and files from it %
inds = hidden_indices(data_dir);
data_dir(inds) = [];
inds = hidden_indices(features_dir);
features_dir(inds) = [];


%% extract the dense-features and save them as .txt file %%
% tic
% fprintf('Starting to extract the Dense-trajectory features for %s dataset. This might take a while.\n', datasetName);
% for class=1:length(data_dir)
%     class_dir_path = fullfile(dataset_path, '/', data_dir(class).name, '/');
%     class_dir = dir(fullfile(class_dir_path, '*.avi'));
%     num_videos = length(class_dir);
%     
%     % change to the class directory %
%     cd(class_dir_path);
%     % extract the features of each video and store them %
%     ! ../../../../lib/run_dense.sh
% end
% toc


%% convert .txt files to .mat files and delete the .txt files %%
% tic
% fprintf('Starting to convert the .txt files to .mat files. This might take a while.\n');
% for class=1%:length(data_dir)
%     class_dir_path = fullfile(features_path, '/', data_dir(class).name, '/');
%     class_dir = dir(fullfile(class_dir_path, '*.txt'));
%     num_videos = length(class_dir);
%     
%     % change to the class directory %
%     cd(class_dir_path);
% 
%     % convert them to .mat files %
%     parfor i=1:num_videos
% 		features_table = readtable(fullfile(class_dir_path, class_dir(i).name));
% 		features_array = table2array(features_table);
% 		filename = regexprep(class_dir(i).name, '.avi.txt', '.mat');
% 		
%         disp(filename);
% 		parsave(filename, features_array);
%     end
%     
%     % remove the .txt files %
%     fprintf('Removing unnecessary .txt files...\n');
%     ! find ./ -type f -name "*.txt" -delete
%     fprintf('Done\n');
% end
% 
% % get back to the current folder %
% cd(currentFolder);
% toc


%% extract motion maps for each video %%
tic
fprintf('Generating motion maps for %s dataset. This might take a while.\n', datasetName);
generate_motion_maps(dataset_path, features_path);
toc
