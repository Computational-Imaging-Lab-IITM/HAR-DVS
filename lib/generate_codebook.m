function generate_codebook(K, datasetName, features_path, encoded_data_path, currentFolder)
    %% load the features directory %%
    features_dir = dir(features_path);
    % remove the hidden files from it %
    inds = hidden_indices(features_dir);
    features_dir(inds) = [];

    
    %% setting the string for test group name %%
    if strcmp(datasetName, 'UCF11') == 1
        % UCF11 dataset %
        group_set_for_test = strcat('_g', sprintf('%.2d', K), '_');
    elseif strcmp(datasetName, 'IITM_DVS_10') == 1
        % IITM_DVS_10 dataset %
        group_set_for_test = strcat('_person', sprintf('%.2d', K), '_');
    elseif strcmp(datasetName, 'IITM_DVS_12') == 1
        % IITM_DVS_12 dataset %
        group_set_for_test = strcat('_person', sprintf('%.2d', K), '_');
    else
        % unknown dataset %
        group_set_for_test = '';
    end


    %% number of videos in each class %%
    num_video_per_class = [];
    
    for class=1:length(features_dir)
        fprintf('Class = %d\n', class);
        videos = dir(fullfile(features_path, features_dir(class).name, '/dense_features_*.mat'));
        % find the test indices and remove them %
        test_indices = groupfile_indices(videos, group_set_for_test);
        videos(test_indices) = [];

        num_video_per_class(class) = length(videos);
    end
    total_num_videos = sum(num_video_per_class);
    num_features_each_video = ceil(100000/total_num_videos);
    fprintf('\n');
    

    %% gather features for building codebook %%
    tic
    train_features_MBHx = [];
    train_features_MBHy = [];
    train_features_MBHxy = [];
    
    fprintf('Gathering features for codebook...\n');
    for class=1:length(features_dir)
        features_dir_mat = dir(fullfile(features_path, features_dir(class).name, '/dense_features_*.mat'));
        % remove test features %
        test_indices = groupfile_indices(features_dir_mat, group_set_for_test);
        features_dir_mat(test_indices) = [];

        num_videos = length(features_dir_mat);
        class_name = features_dir(class).name;

        features_MBHx = zeros(num_videos, num_features_each_video, 96);
        features_MBHy = zeros(num_videos, num_features_each_video, 96);
        features_MBHxy = zeros(num_videos, num_features_each_video, 192);
        
        parfor i=1:num_videos
            features = load(fullfile(features_path, class_name, features_dir_mat(i).name));
            features = features.var;

            % select 'num_features_each_video' random features from each video
            rng(1); % for consistency
            random_indices = randperm(size(features, 1));
            random_indices = random_indices(1:num_features_each_video);

            % selecting the features %
            features_MBHx(i, :, :) = features(random_indices, 245:340);
            features_MBHy(i, :, :) = features(random_indices, 341:436);
            features_MBHxy(i, :, :) = features(random_indices, 245:436);

            fprintf('Loaded %d random features from video = %d of class = %d\n', num_features_each_video, i, class);
        end
        % adding them to the set %
        fprintf('Adding them to set...');
        
        size_x = size(features_MBHx);
        size_y = size(features_MBHy);
        size_xy = size(features_MBHxy);
        
        X = reshape(features_MBHx, [size_x(1)*size_x(2), size_x(3)]);
        Y = reshape(features_MBHy, [size_y(1)*size_y(2), size_y(3)]);
        XY = reshape(features_MBHxy, [size_xy(1)*size_xy(2), size_xy(3)]);

        % remove rows with all zeros %
        X(find(sum(abs(X), 2)) == 0, :) = [];
        Y(find(sum(abs(Y), 2)) == 0, :) = [];
        XY(find(sum(abs(XY), 2)) == 0, :) = [];
    
        train_features_MBHx = [train_features_MBHx; X];
        train_features_MBHy = [train_features_MBHy; Y];
        train_features_MBHxy = [train_features_MBHxy; XY];
        
        fprintf('Done\n');
    end
    fprintf('Done gathering features\n');
    toc


    %% generate the codebook %%
    num_visual_words = 500;
    fprintf('Generating codebook of length %d words...\n', num_visual_words);

    tic
    vocabulary_MBHx = vision.internal.approximateKMeans(single(train_features_MBHx), num_visual_words);
    toc
    tic
    vocabulary_MBHy = vision.internal.approximateKMeans(single(train_features_MBHy), num_visual_words);
    toc
    tic
    vocabulary_MBHxy = vision.internal.approximateKMeans(single(train_features_MBHxy), num_visual_words);
    toc
    fprintf('Done\n\n');

    fprintf('Saving to matfiles...');
    save('./workspace_data/Codebook_MBHx.mat', 'vocabulary_MBHx', '-v7.3');
    save('./workspace_data/Codebook_MBHy.mat', 'vocabulary_MBHy', '-v7.3');
    save('./workspace_data/Codebook_MBHxy.mat', 'vocabulary_MBHxy', '-v7.3');
    fprintf('Done\n\n');

    %% encode all the features %%
    tic
    % initialize %
    vocab_search_tree_MBHx = vision.internal.Kdtree();
    vocab_search_tree_MBHy = vision.internal.Kdtree();
    vocab_search_tree_MBHxy = vision.internal.Kdtree();

    % indexing %
    index(vocab_search_tree_MBHx, vocabulary_MBHx);
    index(vocab_search_tree_MBHy, vocabulary_MBHy);
    index(vocab_search_tree_MBHxy, vocabulary_MBHxy);
    toc

    %% encoding train features %%
    % delete previously encoded features %
    cd(encoded_data_path);
    ! find ./train/ -type f -name "*.mat" -delete
    cd(currentFolder);
    
    tic
    for i=1:length(features_dir)
        features_dir_mat = dir(fullfile(features_path, features_dir(i).name, '/dense_features_*.mat'));
        % remove test features %
        test_indices = groupfile_indices(features_dir_mat, group_set_for_test);
        features_dir_mat(test_indices) = [];

        num_video_features = length(features_dir_mat);
        class_name = features_dir(i).name;

%         cd(strcat('/media/data/bimal/chinni/MBH_SVM/data/encoded_UCF11/', features_dir(i).name));
        parfor j=1:num_video_features
            A = load(fullfile(features_path, class_name, features_dir_mat(j).name));
            A = A.var;
            A_x = A(:, 245:340);
            A_y = A(:, 341:436);
            A_xy = A(:, 245:436);

            % indexing %
            index(vocab_search_tree_MBHx, vocabulary_MBHx);
            index(vocab_search_tree_MBHy, vocabulary_MBHy);
            index(vocab_search_tree_MBHxy, vocabulary_MBHxy);

            matchIndex_MBHx = vocab_search_tree_MBHx.knnSearch(single(A_x), 1);
            matchIndex_MBHy = vocab_search_tree_MBHy.knnSearch(single(A_y), 1);
            matchIndex_MBHxy = vocab_search_tree_MBHxy.knnSearch(single(A_xy), 1);

            encoded_MBHx = bov_encode(matchIndex_MBHx, num_visual_words);
            encoded_MBHy = bov_encode(matchIndex_MBHy, num_visual_words);
            encoded_MBHxy = bov_encode(matchIndex_MBHxy, num_visual_words);
            
            % output folder name %
            foldername = fullfile(encoded_data_path, 'train/', features_dir(i).name, '/');
            % check if the folder exists, else creae one %
            if exist(foldername, 'dir') ~= 7
                % directory doesn't exist so create it %
                fprintf('Directory %s does not exist. So creating it...', foldername);
                mkdir(foldername);
                fprintf('Done\n');
            end
            % output filename %
            filename = strcat(foldername, 'encoded_', features_dir_mat(j).name);
            
            filename_MBHx = regexprep(filename, '.mat', '_MBHx.mat');
            filename_MBHy = regexprep(filename, '.mat', '_MBHy.mat');
            filename_MBHxy = regexprep(filename, '.mat', '_MBHxy.mat');

            parsave(filename_MBHx, encoded_MBHx);
            parsave(filename_MBHy, encoded_MBHy);
            parsave(filename_MBHxy, encoded_MBHxy);
            
            disp(features_dir_mat(j).name);
        end
    end
    toc

    
    %% encoding test features %%
    % delete previously encoded features %
    cd(encoded_data_path);
    ! find ./test/ -type f -name "*.mat" -delete
    cd(currentFolder);
    
    tic
    data_dir_test = dir(features_path);
    inds = hidden_indices(data_dir_test);
    data_dir_test(inds) = [];
    
    for i=1:length(data_dir_test)
        features_dir_mat = dir(fullfile(features_path, data_dir_test(i).name, '/dense_features_*.mat'));
        % take only test features %
        test_indices = groupfile_indices(features_dir_mat, group_set_for_test);
        features_dir_mat = features_dir_mat(test_indices);

        num_video_features = length(features_dir_mat);
        class_name = data_dir_test(i).name;

%         cd(strcat('/media/data/bimal/chinni/MBH_SVM/data/test_encoded_UCF11/', data_dir_test(i).name));
        parfor j=1:num_video_features
            A = load(fullfile(features_path, class_name, features_dir_mat(j).name));
            A = A.var;
            A_x = A(:, 245:340);
            A_y = A(:, 341:436);
            A_xy = A(:, 245:436);

            % indexing %
            index(vocab_search_tree_MBHx, vocabulary_MBHx);
            index(vocab_search_tree_MBHy, vocabulary_MBHy);
            index(vocab_search_tree_MBHxy, vocabulary_MBHxy);

            matchIndex_MBHx = vocab_search_tree_MBHx.knnSearch(single(A_x), 1);
            matchIndex_MBHy = vocab_search_tree_MBHy.knnSearch(single(A_y), 1);
            matchIndex_MBHxy = vocab_search_tree_MBHxy.knnSearch(single(A_xy), 1);

            encoded_MBHx = bov_encode(matchIndex_MBHx, num_visual_words);
            encoded_MBHy = bov_encode(matchIndex_MBHy, num_visual_words);
            encoded_MBHxy = bov_encode(matchIndex_MBHxy, num_visual_words);

            % output folder name %
            foldername = fullfile(encoded_data_path, 'test/', features_dir(i).name, '/');
            % check if the folder exists, else creae one %
            if exist(foldername, 'dir') ~= 7 
                % directory doesn't exist so create it %
                fprintf('Directory %s does not exist. So creating it...', foldername);
                mkdir(foldername);
                fprintf('Done\n');
            end
            % output filename %
            filename = strcat(foldername, 'encoded_', features_dir_mat(j).name);
            
            filename_MBHx = regexprep(filename, '.mat', '_MBHx.mat');
            filename_MBHy = regexprep(filename, '.mat', '_MBHy.mat');
            filename_MBHxy = regexprep(filename, '.mat', '_MBHxy.mat');

            parsave(filename_MBHx, encoded_MBHx);
            parsave(filename_MBHy, encoded_MBHy);
            parsave(filename_MBHxy, encoded_MBHxy);
            
            disp(features_dir_mat(j).name);
        end
    end
    toc

    
end



