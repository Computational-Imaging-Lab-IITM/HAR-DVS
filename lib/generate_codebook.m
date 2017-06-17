function generate_codebook(K, datasetName, features_path, encoded_data_path, currentFolder)
    %% load the features directory %%
    features_dir = dir(features_path);
    % remove the hidden files from it %
    inds = hidden_indices(features_dir);
    features_dir(inds) = [];
    
    
    %% setting the string for test group name %%
    if strcmp(datasetName, 'UCF11_DVS') == 1
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
    
    
    %% gather motion-maps for building SURF codebook %%
    tic

    % extension of images for motion maps %
    exts = {'.jpg'};
    
    fprintf('Gathering motion-maps for codebook...');
    
    % motion maps %
    imds_mm_train_xy = imageDatastore(features_path, 'IncludeSubfolders', true, 'FileExtensions', exts);
    imds_mm_train_xt = imageDatastore(features_path, 'IncludeSubfolders', true, 'FileExtensions', exts);
    imds_mm_train_yt = imageDatastore(features_path, 'IncludeSubfolders', true, 'FileExtensions', exts);
    
    % remove test set %
    test_indices_mm = groupfile_indices_mm(imds_mm_train_xy.Files, group_set_for_test);
    imds_mm_train_xy.Files(test_indices_mm, :) = [];
    imds_mm_train_xt.Files(test_indices_mm, :) = [];
    imds_mm_train_yt.Files(test_indices_mm, :) = [];
    
    % take xy, xt, yt into respective imds objects %
    indices_xy = groupfile_indices_mm(imds_mm_train_xy.Files, '_xy.jpg');
    indices_xt = groupfile_indices_mm(imds_mm_train_xt.Files, '_xt.jpg');
    indices_yt = groupfile_indices_mm(imds_mm_train_yt.Files, '_yt.jpg');
    imds_mm_train_xy.Files = imds_mm_train_xy.Files(indices_xy, :);
    imds_mm_train_xt.Files = imds_mm_train_xt.Files(indices_xt, :);
    imds_mm_train_yt.Files = imds_mm_train_yt.Files(indices_yt, :);
    
    fprintf('Done\n\n');
    toc
    
    
    %% building BoF codebook for motion-maps %%
    fprintf('Building codebok for motion-maps. This will take a while.\n');
    tic
    bag_xy = bagOfFeatures(imds_mm_train_xy);
    toc
    tic
    bag_xt = bagOfFeatures(imds_mm_train_xt);
    toc
    tic
    bag_yt = bagOfFeatures(imds_mm_train_yt);
    toc
    fprintf('Done\n\n');
    
    fprintf('Saving to matfiles...');
    save('./workspace_data/Codebook_MMxy.mat', 'bag_xy', '-v7.3');
    save('./workspace_data/Codebook_MMxt.mat', 'bag_xt', '-v7.3');
    save('./workspace_data/Codebook_MMyt.mat', 'bag_yt', '-v7.3');
    fprintf('Done\n\n');

    %% gather features for building codebook %%
    tic
    train_features_HoG = [];
    train_features_HoF = [];
    train_features_MBHx = [];
    train_features_MBHy = [];
    
    fprintf('Gathering dense features for codebook...\n');
    for class=1:length(features_dir)
        % dense features %
        features_dir_mat = dir(fullfile(features_path, features_dir(class).name, '/dense_features_*.mat'));
        % remove test features %
        test_indices = groupfile_indices(features_dir_mat, group_set_for_test);
        features_dir_mat(test_indices) = [];
        
        % number of videos in class %
        num_videos = length(features_dir_mat);
        class_name = features_dir(class).name;
        
        % initializse the features matrices %
        features_HoG = zeros(num_videos, num_features_each_video, 96);
        features_HoF = zeros(num_videos, num_features_each_video, 108);
        features_MBHx = zeros(num_videos, num_features_each_video, 96);
        features_MBHy = zeros(num_videos, num_features_each_video, 96);
        
        parfor i=1:num_videos
            features = load(fullfile(features_path, class_name, features_dir_mat(i).name));
            features = features.var;
            
            % appending zero rows if the video has less features %
            feat_size = size(features, 1);
            if feat_size < num_features_each_video
                features(feat_size+1:num_features_each_video, :) = zeros(num_features_each_video-feat_size, 437);
            end
            
            % select 'num_features_each_video' random features from each video
            rng(1); % for consistency
            random_indices = randperm(size(features, 1));
            random_indices = random_indices(1:num_features_each_video);

            % selecting the features %
            features_HoG(i, :, :) = features(random_indices, 41:136);
            features_HoF(i, :, :) = features(random_indices, 137:244);
            features_MBHx(i, :, :) = features(random_indices, 245:340);
            features_MBHy(i, :, :) = features(random_indices, 341:436);

            fprintf('Loaded %d random features from video = %d of class = %d\n', num_features_each_video, i, class);
        end
        % adding them to the set %
        fprintf('Adding them to set...');
        
        size_hog = size(features_HoG);
        size_hof = size(features_HoF);
        size_x = size(features_MBHx);
        size_y = size(features_MBHy);
        
        HoG = reshape(features_HoG, [size_hog(1)*size_hog(2), size_hog(3)]);
        HoF = reshape(features_HoF, [size_hof(1)*size_hof(2), size_hof(3)]);        
        MBHx = reshape(features_MBHx, [size_x(1)*size_x(2), size_x(3)]);
        MBHy = reshape(features_MBHy, [size_y(1)*size_y(2), size_y(3)]);

        % remove rows with all zeros %
        HoG(find(sum(abs(HoG), 2)) == 0, :) = [];
        HoF(find(sum(abs(HoF), 2)) == 0, :) = [];
        MBHx(find(sum(abs(MBHx), 2)) == 0, :) = [];
        MBHy(find(sum(abs(MBHy), 2)) == 0, :) = [];
        
        train_features_HoG = [train_features_HoG; HoG];
        train_features_HoF = [train_features_HoF; HoF];
        train_features_MBHx = [train_features_MBHx; MBHx];
        train_features_MBHy = [train_features_MBHy; MBHy];
        
        fprintf('Done\n');
    end
    fprintf('Done gathering features\n');
    toc


    %% generate the codebook %%
    num_visual_words = 500;
    fprintf('Generating codebook of length %d words...\n', num_visual_words);
    
    tic
    vocabulary_HoG = vision.internal.approximateKMeans(single(train_features_HoG), num_visual_words);
    toc
    tic
    vocabulary_HoF = vision.internal.approximateKMeans(single(train_features_HoF), num_visual_words);
    toc
    tic
    vocabulary_MBHx = vision.internal.approximateKMeans(single(train_features_MBHx), num_visual_words);
    toc
    tic
    vocabulary_MBHy = vision.internal.approximateKMeans(single(train_features_MBHy), num_visual_words);
    toc
    fprintf('Done\n\n');

    fprintf('Saving to matfiles...');
    save('./workspace_data/Codebook_HoG.mat', 'vocabulary_HoG', '-v7.3');
    save('./workspace_data/Codebook_HoF.mat', 'vocabulary_HoF', '-v7.3');
    save('./workspace_data/Codebook_MBHx.mat', 'vocabulary_MBHx', '-v7.3');
    save('./workspace_data/Codebook_MBHy.mat', 'vocabulary_MBHy', '-v7.3');
    fprintf('Done\n\n');
    

    %% encode all the features %%
    tic
    % initialize %
    vocab_search_tree_HoG = vision.internal.Kdtree();
    vocab_search_tree_HoF = vision.internal.Kdtree();
    vocab_search_tree_MBHx = vision.internal.Kdtree();
    vocab_search_tree_MBHy = vision.internal.Kdtree();

    % indexing %
    index(vocab_search_tree_HoG, vocabulary_HoG);
    index(vocab_search_tree_HoF, vocabulary_HoF);
    index(vocab_search_tree_MBHx, vocabulary_MBHx);
    index(vocab_search_tree_MBHy, vocabulary_MBHy);
    toc
    

    %% encoding train features %%
    % delete previously encoded features %
    cd(encoded_data_path);
    ! find ./train/ -type f -name "*.mat" -delete
    cd(currentFolder);
    
    tic
    for i=1:length(features_dir)
        features_dir_mat = dir(fullfile(features_path, features_dir(i).name, '/dense_features_*.mat'));
        motion_maps_xy = dir(fullfile(features_path, features_dir(i).name, '/*_xy.jpg'));
        motion_maps_xt = dir(fullfile(features_path, features_dir(i).name, '/*_xt.jpg'));
        motion_maps_yt = dir(fullfile(features_path, features_dir(i).name, '/*_yt.jpg'));
        % remove test features %
        test_indices = groupfile_indices(features_dir_mat, group_set_for_test);
        features_dir_mat(test_indices) = [];
        motion_maps_xy(test_indices) = [];
        motion_maps_xt(test_indices) = [];
        motion_maps_yt(test_indices) = [];
        
        % number of videos %
        num_video_features = length(features_dir_mat);
        class_name = features_dir(i).name;

%         cd(strcat('/media/data/bimal/chinni/MBH_SVM/data/encoded_UCF11/', features_dir(i).name));
        parfor j=1:num_video_features
            % loading dense features %
            A = load(fullfile(features_path, class_name, features_dir_mat(j).name));
            A = A.var;
            if size(A, 1) ~= 0
                A_hog = A(:, 41:136);
                A_hof = A(:, 137:244);
                A_x = A(:, 245:340);
                A_y = A(:, 341:436);
                % loading motion-maps images %
                mm_img_xy = imread(fullfile(features_path, class_name, motion_maps_xy(j).name));
                mm_img_xt = imread(fullfile(features_path, class_name, motion_maps_xt(j).name));
                mm_img_yt = imread(fullfile(features_path, class_name, motion_maps_yt(j).name));

                % indexing %
                index(vocab_search_tree_HoG, vocabulary_HoG);
                index(vocab_search_tree_HoF, vocabulary_HoF);
                index(vocab_search_tree_MBHx, vocabulary_MBHx);
                index(vocab_search_tree_MBHy, vocabulary_MBHy);

                matchIndex_HoG = vocab_search_tree_HoG.knnSearch(single(A_hog), 1);
                matchIndex_HoF = vocab_search_tree_HoF.knnSearch(single(A_hof), 1);
                matchIndex_MBHx = vocab_search_tree_MBHx.knnSearch(single(A_x), 1);
                matchIndex_MBHy = vocab_search_tree_MBHy.knnSearch(single(A_y), 1);

                % encoding dense features %
                encoded_HoG = bov_encode(matchIndex_HoG, num_visual_words);
                encoded_HoF = bov_encode(matchIndex_HoF, num_visual_words);
                encoded_MBHx = bov_encode(matchIndex_MBHx, num_visual_words);
                encoded_MBHy = bov_encode(matchIndex_MBHy, num_visual_words);
                % encoding motion maps %
                encoded_xy = encode(bag_xy, mm_img_xy);
                encoded_xt = encode(bag_xt, mm_img_xt);
                encoded_yt = encode(bag_yt, mm_img_yt);

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
                filename_mm_xy = strcat(foldername, 'encoded_', motion_maps_xy(j).name);
                filename_mm_xt = strcat(foldername, 'encoded_', motion_maps_xt(j).name);
                filename_mm_yt = strcat(foldername, 'encoded_', motion_maps_yt(j).name);

                filename_HoG = regexprep(filename, '.mat', '_HoG.mat');
                filename_HoF = regexprep(filename, '.mat', '_HoF.mat');
                filename_MBHx = regexprep(filename, '.mat', '_MBHx.mat');
                filename_MBHy = regexprep(filename, '.mat', '_MBHy.mat');
                filename_MMxy = regexprep(filename_mm_xy, '.jpg', '.mat'); 
                filename_MMxt = regexprep(filename_mm_xt, '.jpg', '.mat'); 
                filename_MMyt = regexprep(filename_mm_yt, '.jpg', '.mat'); 

                parsave(filename_HoG, encoded_HoG);
                parsave(filename_HoF, encoded_HoF);
                parsave(filename_MBHx, encoded_MBHx);
                parsave(filename_MBHy, encoded_MBHy);
                parsave(filename_MMxy, encoded_xy);
                parsave(filename_MMxt, encoded_xt);
                parsave(filename_MMyt, encoded_yt);

                disp(features_dir_mat(j).name);
            end
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
        motion_maps_xy = dir(fullfile(features_path, features_dir(i).name, '/*_xy.jpg'));
        motion_maps_xt = dir(fullfile(features_path, features_dir(i).name, '/*_xt.jpg'));
        motion_maps_yt = dir(fullfile(features_path, features_dir(i).name, '/*_yt.jpg'));
        % take only test features %
        test_indices = groupfile_indices(features_dir_mat, group_set_for_test);
        features_dir_mat = features_dir_mat(test_indices);
        motion_maps_xy = motion_maps_xy(test_indices);
        motion_maps_xt = motion_maps_xt(test_indices);
        motion_maps_yt = motion_maps_yt(test_indices);

        num_video_features = length(features_dir_mat);
        class_name = data_dir_test(i).name;

%         cd(strcat('/media/data/bimal/chinni/MBH_SVM/data/test_encoded_UCF11/', data_dir_test(i).name));
        parfor j=1:num_video_features
            % loading dense features %
            A = load(fullfile(features_path, class_name, features_dir_mat(j).name));
            A = A.var;
            if size(A, 1) ~= 0
                A_hog = A(:, 41:136);
                A_hof = A(:, 137:244);
                A_x = A(:, 245:340);
                A_y = A(:, 341:436);
                % loading motion-maps images %
                mm_img_xy = imread(fullfile(features_path, class_name, motion_maps_xy(j).name));
                mm_img_xt = imread(fullfile(features_path, class_name, motion_maps_xt(j).name));
                mm_img_yt = imread(fullfile(features_path, class_name, motion_maps_yt(j).name));

                % indexing %
                index(vocab_search_tree_HoG, vocabulary_HoG);
                index(vocab_search_tree_HoF, vocabulary_HoF);
                index(vocab_search_tree_MBHx, vocabulary_MBHx);
                index(vocab_search_tree_MBHy, vocabulary_MBHy);

                matchIndex_HoG = vocab_search_tree_HoG.knnSearch(single(A_hog), 1);
                matchIndex_HoF = vocab_search_tree_HoF.knnSearch(single(A_hof), 1);
                matchIndex_MBHx = vocab_search_tree_MBHx.knnSearch(single(A_x), 1);
                matchIndex_MBHy = vocab_search_tree_MBHy.knnSearch(single(A_y), 1);

                % encoding dense features %
                encoded_HoG = bov_encode(matchIndex_HoG, num_visual_words);
                encoded_HoF = bov_encode(matchIndex_HoF, num_visual_words);
                encoded_MBHx = bov_encode(matchIndex_MBHx, num_visual_words);
                encoded_MBHy = bov_encode(matchIndex_MBHy, num_visual_words);
                % encoding motion maps %
                encoded_xy = encode(bag_xy, mm_img_xy);
                encoded_xt = encode(bag_xt, mm_img_xt);
                encoded_yt = encode(bag_yt, mm_img_yt);

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
                filename_mm_xy = strcat(foldername, 'encoded_', motion_maps_xy(j).name);
                filename_mm_xt = strcat(foldername, 'encoded_', motion_maps_xt(j).name);
                filename_mm_yt = strcat(foldername, 'encoded_', motion_maps_yt(j).name);

                filename_HoG = regexprep(filename, '.mat', '_HoG.mat');
                filename_HoF = regexprep(filename, '.mat', '_HoF.mat');
                filename_MBHx = regexprep(filename, '.mat', '_MBHx.mat');
                filename_MBHy = regexprep(filename, '.mat', '_MBHy.mat');
                filename_MMxy = regexprep(filename_mm_xy, '.jpg', '.mat'); 
                filename_MMxt = regexprep(filename_mm_xt, '.jpg', '.mat'); 
                filename_MMyt = regexprep(filename_mm_yt, '.jpg', '.mat'); 

                parsave(filename_HoG, encoded_HoG);
                parsave(filename_HoF, encoded_HoF);
                parsave(filename_MBHx, encoded_MBHx);
                parsave(filename_MBHy, encoded_MBHy);
                parsave(filename_MMxy, encoded_xy);
                parsave(filename_MMxt, encoded_xt);
                parsave(filename_MMyt, encoded_yt);

                disp(features_dir_mat(j).name);
            end
        end
    end
    toc

    
end



