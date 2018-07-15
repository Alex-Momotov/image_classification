%% Reduce dimensionality with PCA
% We can rerun this part whenever we want to choose different number of
% features for our classifiers later on. The old data file will be
% overwritten with new data with specified number of features.

% Clear workspace
clear
close all
clc

% Load computed features and labels
load('CW2Data.mat','trn_features','tst_features','trnLabel','tstLabel');

% Mean centering data
trn_means = mean(trn_features);
trn_mean_cent = trn_features - trn_means;
tst_mean_cent = tst_features - trn_means;

% Singular value decomposition
[U, S, V] = svd(trn_mean_cent);

% Explore eigen values to decide how many to keep
for i=1:324
    disp(i + " " + S(i,i));
end

% Project data to the PCA space, then reduce it
num_feats = 55;
trn_proj = trn_mean_cent/V';
tst_proj = tst_mean_cent/V';
trn_proj = trn_proj(:,1:num_feats);
tst_proj = tst_proj(:,1:num_feats);

% Save reduced data for later classification
save('Reduced_data_PCA.mat','trn_proj','tst_proj');

%% Reduce dimensionality with LDA
% This part uses Fisher's Linear Discriminant Derivation which is the
% supervised dimensionality reduction part of LDA. To run, you need to
% install "FDA LDA multiclass" Add-On by Sultan Alzahrani.

% Clear workspace
clear
close all
clc

% Load computed features and labels
load('CW2Data.mat','trn_features','tst_features','trnLabel','tstLabel');

% Fisher's Linear Discriminant Derivation and data reduction
% Z is transposed reduced data, W is projection matrix
num_feats = 9;
[Z,W] = FDA(trn_features',trnLabel,num_feats);

% Orientate reduced tranining data correctly and reduce testing data
trn_proj = Z';
tst_proj = tst_features * W;

% Save reduced data for later classification
save('Reduced_data_LDA.mat','trn_proj','tst_proj');

%% Reduce dimensionality with PCA, then further reduce it with LDA
% This part loads data reduced with PCA to 55 features, then further
% reduces it with LDA to 9 features. In classification stages we show how 
% data reduced with PCA then LDA outperforms data reduced with PCA-only or
% LDA-only for some classifiers.

% Clear workspace
clear
close all
clc

% Load PCA reduced data and labels
load('Reduced_data_PCA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Fisher's Linear Discriminant Derivation and data reduction
% Z is transposed reduced data, W is projection matrix
num_feats = 9;
[Z,W] = FDA(trn_proj',trnLabel,num_feats);

% Orientate reduced tranining data correctly and reduce testing data
trn_proj = Z';
tst_proj = tst_proj * W;

% Save reduced data for later classification
save('Reduced_data_PCA_and_LDA.mat','trn_proj','tst_proj');
