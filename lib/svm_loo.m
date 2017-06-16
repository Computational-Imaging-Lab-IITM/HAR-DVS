function [accuracy_xy_xt_yt, accuracy_xy_xt_yt_MBHx_MBHy, accuracy_xy_xt_yt_HoF, accuracy_HoG, accuracy_HoF, accuracy_MBHx_MBHy, accuracy_xy, accuracy_xt, accuracy_yt, confusion_xy_xt_yt, confusion_xy_xt_yt_MBHx_MBHy, confusion_xy_xt_yt_HoF, confusion_HoG, confusion_HoF, confusion_MBHx_MBHy, confusion_xy, confusion_xt, confusion_yt] = svm_loo(encoded_data_path, num_classes)
    %% data directories %%
    train_enc = fullfile(encoded_data_path, 'train/');
    train_data_dir = dir(train_enc);

    test_enc = fullfile(encoded_data_path, 'test/');
    test_data_dir = dir(test_enc);

    % remove the hidden files from the folders %
    inds = hidden_indices(train_data_dir);
    train_data_dir(inds) = [];

    inds = hidden_indices(test_data_dir);
    test_data_dir(inds) = [];

    
    %% load all the train features %%
    train_features_HoG = [];
    train_features_HoF = [];
    train_features_MBHx = [];
    train_features_MBHy = [];
    train_features_MMxy = [];
    train_features_MMxt = [];
    train_features_MMyt = [];

    train_labels = [];

    tic
    for class=1:length(train_data_dir)
        % dense features %
        features_dir_HoG = dir(fullfile(train_enc, train_data_dir(class).name, '/*_HoG.mat'));
        features_dir_HoF = dir(fullfile(train_enc, train_data_dir(class).name, '/*_HoF.mat'));
        features_dir_MBHx = dir(fullfile(train_enc, train_data_dir(class).name, '/*_MBHx.mat'));
        features_dir_MBHy = dir(fullfile(train_enc, train_data_dir(class).name, '/*_MBHy.mat'));
        % motion-maps features %
        features_dir_MMxy = dir(fullfile(train_enc, train_data_dir(class).name, '/*_xy.mat'));
        features_dir_MMxt = dir(fullfile(train_enc, train_data_dir(class).name, '/*_xt.mat'));
        features_dir_MMyt = dir(fullfile(train_enc, train_data_dir(class).name, '/*_yt.mat'));

        fprintf('Updating train labels for class = %d...', class);
        train_labels = [train_labels, class*ones(1, length(features_dir_MBHx))];
        fprintf('Done\n\n');

        fprintf('Preparing to load training features of class = %d\n', class);
        
        offset = size(train_features_MBHx, 1);
        parfor i=1:length(features_dir_MBHx)
            % dense features %
            fHoG = load(fullfile(train_enc, train_data_dir(class).name, features_dir_HoG(i).name));
            fHoF = load(fullfile(train_enc, train_data_dir(class).name, features_dir_HoF(i).name));
            fMBHx = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MBHx(i).name));
            fMBHy = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MBHy(i).name));
            % motion-maps features %
            fMMxy = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MMxy(i).name));
            fMMxt = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MMxt(i).name));
            fMMyt = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MMyt(i).name));
            
            fHoG = fHoG.var;
            fHoF = fHoF.var;
            fMBHx = fMBHx.var;
            fMBHy = fMBHy.var;
            fMMxy = fMMxy.var;
            fMMxt = fMMxt.var;
            fMMyt = fMMyt.var;
            
            train_features_HoG(offset + i, :) = fHoG(1, :);
            train_features_HoF(offset + i, :) = fHoF(1, :);
            train_features_MBHx(offset + i, :) = fMBHx(1, :);
            train_features_MBHy(offset + i, :) = fMBHy(1, :);
            train_features_MMxy(offset + i, :) = fMMxy(1, :);
            train_features_MMxt(offset + i, :) = fMMxt(1, :);
            train_features_MMyt(offset + i, :) = fMMyt(1, :);

            fprintf('Loaded train data for class = %d and video = %d\n', class, i);
        end
        fprintf('Done\n');
    end
    toc
    

    %% load all the test features %%
    test_features_HoG = [];
    test_features_HoF = [];
    test_features_MBHx = [];
    test_features_MBHy = [];
    test_features_MMxy = [];
    test_features_MMxt = [];
    test_features_MMyt = [];

    test_labels = [];

    tic
    for class=1:length(test_data_dir) % number of classes
        % dense features %
        features_dir_HoG = dir(fullfile(test_enc, test_data_dir(class).name, '/*_HoG.mat'));
        features_dir_HoF = dir(fullfile(test_enc, test_data_dir(class).name, '/*_HoF.mat'));
        features_dir_MBHx = dir(fullfile(test_enc, test_data_dir(class).name, '/*_MBHx.mat'));
        features_dir_MBHy = dir(fullfile(test_enc, test_data_dir(class).name, '/*_MBHy.mat'));
        % motion-maps features %
        features_dir_MMxy = dir(fullfile(test_enc, test_data_dir(class).name, '/*_xy.mat'));
        features_dir_MMxt = dir(fullfile(test_enc, test_data_dir(class).name, '/*_xt.mat'));
        features_dir_MMyt = dir(fullfile(test_enc, test_data_dir(class).name, '/*_yt.mat'));

        fprintf('Updating test labels for class = %d...', class);
        test_labels = [test_labels, class*ones(1, length(features_dir_MBHx))];
        fprintf('Done\n\n');

        fprintf('Preparing to load test features of class = %d\n', class);

        offset = size(test_features_MBHx, 1);
        parfor i=1:length(features_dir_MBHx)
            % dense features %
            fHoG = load(fullfile(test_enc, test_data_dir(class).name, features_dir_HoG(i).name));
            fHoF = load(fullfile(test_enc, test_data_dir(class).name, features_dir_HoF(i).name));
            fMBHx = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MBHx(i).name));
            fMBHy = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MBHy(i).name));
            % motion-maps features %
            fMMxy = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MMxy(i).name));
            fMMxt = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MMxt(i).name));
            fMMyt = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MMyt(i).name));
            
            fHoG = fHoG.var;
            fHoF = fHoF.var;
            fMBHx = fMBHx.var;
            fMBHy = fMBHy.var;
            fMMxy = fMMxy.var;
            fMMxt = fMMxt.var;
            fMMyt = fMMyt.var;
            
            test_features_HoG(offset + i, :) = fHoG(1, :);
            test_features_HoF(offset + i, :) = fHoF(1, :);
            test_features_MBHx(offset + i, :) = fMBHx(1, :);
            test_features_MBHy(offset + i, :) = fMBHy(1, :);
            test_features_MMxy(offset + i, :) = fMMxy(1, :);
            test_features_MMxt(offset + i, :) = fMMxt(1, :);
            test_features_MMyt(offset + i, :) = fMMyt(1, :);
            
            fprintf('Loaded test data for class = %d and video = %d\n', class, i);
        end
        fprintf('Done\n');
    end
    toc

    
    %% build SVMs and predict %%
    % building SVMs %
    t = templateSVM('KernelFunction', 'linear');
    fprintf('Building SVMs...\n');
    
    tic
    svmc_xy_xt_yt = fitcecoc([train_features_MMxy, train_features_MMxt, train_features_MMyt], train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_xy_xt_yt_MBHx_MBHy = fitcecoc([train_features_MMxy, train_features_MMxt, train_features_MMyt, train_features_MBHx, train_features_MBHy], train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_xy_xt_yt_HoF = fitcecoc([train_features_MMxy, train_features_MMxt, train_features_MMyt, train_features_HoF], train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_HoG = fitcecoc(train_features_HoG, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_HoF = fitcecoc(train_features_HoF, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_MBHx_MBHy = fitcecoc([train_features_MBHx, train_features_MBHy], train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_xy = fitcecoc(train_features_MMxy, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_xt = fitcecoc(train_features_MMxt, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    tic
    svmc_yt = fitcecoc(train_features_MMyt, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    
    fprintf('Done\n');

    % predicting %
    fprintf('Predicting...');
    
    [pred_xy_xt_yt, score_xy_xt_yt, cost_xy_xt_yt] = predict(svmc_xy_xt_yt, [test_features_MMxy, test_features_MMxt, test_features_MMyt]);
    
    [pred_xy_xt_yt_MBHx_MBHy, score_xy_xt_yt_MBHx_MBHy, cost_xy_xt_yt_MBHx_MBHy] = predict(svmc_xy_xt_yt_MBHx_MBHy, [test_features_MMxy, test_features_MMxt, test_features_MMyt, test_features_MBHx, test_features_MBHy]);
    
    [pred_xy_xt_yt_HoF, score_xy_xt_yt_HoF, cost_xy_xt_yt_HoF] = predict(svmc_xy_xt_yt_HoF, [test_features_MMxy, test_features_MMxt, test_features_MMyt, test_features_HoF]);
    
    [pred_HoG, score_HoG, cost_HoG] = predict(svmc_HoG, test_features_HoG);
    
    [pred_HoF, score_HoF, cost_HoF] = predict(svmc_HoF, test_features_HoF);
    
    [pred_MBHx_MBHy, score_MBHx_MBHy, cost_MBHx_MBHy] = predict(svmc_MBHx_MBHy, [test_features_MBHx, test_features_MBHy]);
    
    [pred_xy, score_xy, cost_xy] = predict(svmc_xy, test_features_MMxy);
    
    [pred_xt, score_xt, cost_xt] = predict(svmc_xt, test_features_MMxt);
    
    [pred_yt, score_yt, cost_yt] = predict(svmc_yt, test_features_MMyt);
    
    fprintf('Done\n');

    % confusion matrices %
    [confusion_xy_xt_yt, order_xy_xt_yt] = confusionmat(test_labels, pred_xy_xt_yt);
    
    [confusion_xy_xt_yt_MBHx_MBHy, order_xy_xt_yt_MBHx_MBHy] = confusionmat(test_labels, pred_xy_xt_yt_MBHx_MBHy);
    
    [confusion_xy_xt_yt_HoF, order_xy_xt_yt_HoF] = confusionmat(test_labels, pred_xy_xt_yt_HoF);
    
    [confusion_HoG, order_HoG] = confusionmat(test_labels, pred_HoG);
    
    [confusion_HoF, order_HoF] = confusionmat(test_labels, pred_HoF);
    
    [confusion_MBHx_MBHy, order_MBHx_MBHy] = confusionmat(test_labels, pred_MBHx_MBHy);
    
    [confusion_xy, order_xy] = confusionmat(test_labels, pred_xy);
    
    [confusion_xt, order_xt] = confusionmat(test_labels, pred_xt);
    
    [confusion_yt, order_yt] = confusionmat(test_labels, pred_yt);

    % normalize confusion matrix %
    confusion_xy_xt_yt = confusion_xy_xt_yt/length(test_labels);
    
    confusion_xy_xt_yt_MBHx_MBHy = confusion_xy_xt_yt_MBHx_MBHy/length(test_labels);
    
    confusion_xy_xt_yt_HoF = confusion_xy_xt_yt_HoF/length(test_labels);
    
    confusion_HoG = confusion_HoG/length(test_labels);
    
    confusion_HoF = confusion_HoF/length(test_labels);
    
    confusion_MBHx_MBHy = confusion_MBHx_MBHy/length(test_labels);
    
    confusion_xy = confusion_xy/length(test_labels);
    
    confusion_xt = confusion_xt/length(test_labels);
    
    confusion_yt = confusion_yt/length(test_labels);

    % accuracies %
    accuracy_xy_xt_yt = trace(confusion_xy_xt_yt)/sum(sum(confusion_xy_xt_yt));
    
    accuracy_xy_xt_yt_MBHx_MBHy = trace(confusion_xy_xt_yt_MBHx_MBHy)/sum(sum(confusion_xy_xt_yt_MBHx_MBHy));
    
    accuracy_xy_xt_yt_HoF = trace(confusion_xy_xt_yt_HoF)/sum(sum(confusion_xy_xt_yt_HoF));
    
    accuracy_HoG = trace(confusion_HoG)/sum(sum(confusion_HoG));
    
    accuracy_HoF = trace(confusion_HoF)/sum(sum(confusion_HoF));
    
    accuracy_MBHx_MBHy = trace(confusion_MBHx_MBHy)/sum(sum(confusion_MBHx_MBHy));
    
    accuracy_xy = trace(confusion_xy)/sum(sum(confusion_xy));
    
    accuracy_xt = trace(confusion_xt)/sum(sum(confusion_xt));
    
    accuracy_yt = trace(confusion_yt)/sum(sum(confusion_yt));
    
    
end