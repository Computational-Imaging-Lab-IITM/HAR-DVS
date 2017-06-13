%% encode the given feature vector %% 
function [featureVector] = bov_encode(matchIndex, VocabularySize)
    % note that lower and upper are both ixcluded %
    featureVector = histcounts(single(matchIndex), 'BinLimits', [1, VocabularySize], 'BinMethod', 'integers');

    % l2 normalisation %
    featureVector = featureVector ./ (norm(featureVector, 2) + eps('single'));
end