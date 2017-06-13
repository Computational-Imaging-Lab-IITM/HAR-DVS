function [accuracy_MBHx, accuracy_MBHy, accuracy_MBHxy, accuracy_MBHxy_rep, confusion_MBHx, confusion_MBHy, confusion_MBHxy, confusion_MBHxy_rep] = svm_loo()
    %% data directories %%
    train_enc = '/media/data/bimal/chinni/MBH_SVM/data/encoded_UCF11/';
    train_data_dir = dir(train_enc);

    test_enc = '/media/data/bimal/chinni/MBH_SVM/data/test_encoded_UCF11/';
    test_data_dir = dir(test_enc);

    % remove the hidden files from the folders %
    inds = hidden_indices(train_data_dir);
    train_data_dir(inds) = [];

    inds = hidden_indices(test_data_dir);
    test_data_dir(inds) = [];

    %% load all the train features %%
    train_features_MBHx = [];
    train_features_MBHy = [];
    train_features_MBHxy = [];

    train_labels = [];

    tic
    for class=1:length(train_data_dir)
        features_dir_MBHx = dir(fullfile(train_enc, train_data_dir(class).name, '/*_MBHx.mat'));
        features_dir_MBHy = dir(fullfile(train_enc, train_data_dir(class).name, '/*_MBHy.mat'));
        features_dir_MBHxy = dir(fullfile(train_enc, train_data_dir(class).name, '/*_MBHxy.mat'));

        fprintf('Updating train labels for class = %d...', class);
        train_labels = [train_labels, class*ones(1, length(features_dir_MBHxy))];
        fprintf('Done\n\n');

        fprintf('Preparing to load training features of class = %d\n', class);

        offset = size(train_features_MBHxy, 1);
        parfor i=1:length(features_dir_MBHxy)
            fMBHx = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MBHx(i).name));
            fMBHy = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MBHy(i).name));
            fMBHxy = load(fullfile(train_enc, train_data_dir(class).name, features_dir_MBHxy(i).name));

            fMBHx = fMBHx.var;
            fMBHy = fMBHy.var;
            fMBHxy = fMBHxy.var;

            train_features_MBHx(offset + i, :) = fMBHx(1, :);
            train_features_MBHy(offset + i, :) = fMBHy(1, :);
            train_features_MBHxy(offset + i, :) = fMBHxy(1, :);

            fprintf('Loaded train data for class = %d and video = %d\n', class, i);
        end
        fprintf('Done\n');
    end
    toc

    %% load all the test features %%
    test_features_MBHx = [];
    test_features_MBHy = [];
    test_features_MBHxy = [];

    test_labels = [];

    tic
    for class=1:length(test_data_dir) % class=1:11
        features_dir_MBHx = dir(fullfile(test_enc, test_data_dir(class).name, '/*_MBHx.mat'));
        features_dir_MBHy = dir(fullfile(test_enc, test_data_dir(class).name, '/*_MBHy.mat'));
        features_dir_MBHxy = dir(fullfile(test_enc, test_data_dir(class).name, '/*_MBHxy.mat'));

        fprintf('Updating test labels for class = %d...', class);
        test_labels = [test_labels, class*ones(1, length(features_dir_MBHxy))];
        fprintf('Done\n\n');

        fprintf('Preparing to load test features of class = %d\n', class);

        offset = size(test_features_MBHxy, 1);
        parfor i=1:length(features_dir_MBHxy)
            fMBHx = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MBHx(i).name));
            fMBHy = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MBHy(i).name));
            fMBHxy = load(fullfile(test_enc, test_data_dir(class).name, features_dir_MBHxy(i).name));

            fMBHx = fMBHx.var;
            fMBHy = fMBHy.var;
            fMBHxy = fMBHxy.var;

            test_features_MBHx(offset + i, :) = fMBHx(1, :);
            test_features_MBHy(offset + i, :) = fMBHy(1, :);
            test_features_MBHxy(offset + i, :) = fMBHxy(1, :);

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
    svm_classifier_MBHx = fitcecoc(train_features_MBHx, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    tic
    svm_classifier_MBHy = fitcecoc(train_features_MBHy, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    tic
    svm_classifier_MBHxy = fitcecoc(train_features_MBHxy, train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    tic
    svm_classifier_MBHxy_rep = fitcecoc([train_features_MBHx, train_features_MBHy], train_labels', 'Coding', 'onevsall', 'Learners', t);
    toc
    fprintf('Done\n');

    % predicting %
    fprintf('Predicting...');
    [pred_MBHx, score_MBHx, cost_MBHx] = predict(svm_classifier_MBHx, test_features_MBHx);
    [pred_MBHy, score_MBHy, cost_MBHy] = predict(svm_classifier_MBHy, test_features_MBHy);
    [pred_MBHxy, score_MBHxy, cost_MBHxy] = predict(svm_classifier_MBHxy, test_features_MBHxy);
    [pred_MBHxy_rep, score_MBHxy_rep, cost_MBHxy_rep] = predict(svm_classifier_MBHxy_rep, [test_features_MBHx, test_features_MBHy]);
    fprintf('Done\n');

    % confusion matrices %
    [confusion_MBHx, order_MBHx] = confusionmat(test_labels, pred_MBHx);
    [confusion_MBHy, order_MBHy] = confusionmat(test_labels, pred_MBHy);
    [confusion_MBHxy, order_MBHxy] = confusionmat(test_labels, pred_MBHxy);
    [confusion_MBHxy_rep, order_MBHxy_rep] = confusionmat(test_labels, pred_MBHxy_rep);

    % normalize confusion matrix %
    confusion_MBHx = confusion_MBHx/length(test_labels);
    confusion_MBHy = confusion_MBHy/length(test_labels);
    confusion_MBHxy = confusion_MBHxy/length(test_labels);
    confusion_MBHxy_rep = confusion_MBHxy_rep/length(test_labels);

    % accuracies %
    accuracy_MBHx = trace(confusion_MBHx)/sum(sum(confusion_MBHx));
    accuracy_MBHy = trace(confusion_MBHy)/sum(sum(confusion_MBHy));
    accuracy_MBHxy = trace(confusion_MBHxy)/sum(sum(confusion_MBHxy));
    accuracy_MBHxy_rep = trace(confusion_MBHxy_rep)/sum(sum(confusion_MBHxy_rep));
    
end