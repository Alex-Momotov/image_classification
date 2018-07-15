%% Use LDA on original 324 computed features (no preliminary data reduction)
% This part deploys LDA classifier on unreduced data. Since LDA
% inherently uses supervised data reduction, we report the accuracy as
% 'accuracy on data reduced with LDA'.

% Clear workspace
clear
close all
clc

% Load computed features and labels
load('CW2Data.mat','trn_features','tst_features','trnLabel','tstLabel');

% Fit data into LDA model, then predict testing data
ldaModel = fitcdiscr(trn_features,trnLabel);
y_tst_predict = predict(ldaModel,tst_features);

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict == tstLabel);
disp("Accuracy on LDA-only reduced data = " + accuracy * 100 + "%");
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'), 'LDA on unreduced data');

%% Using LDA on data reduced with PCA
% This part deploys LDA classifier on PCA reduced data. Since LDA
% inherently uses supervised data reduction, we report the accuracy as
% 'accuracy on data reduced with PCA then LDA'.

% Clear workspace
clear
close all

% Load PCA reduced data and labels
load('Reduced_data_PCA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Fit data into LDA model, then predict testing data
ldaModel = fitcdiscr(trn_proj,trnLabel,'DiscrimType','quadratic');
y_tst_predict = predict(ldaModel,tst_proj);

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict == tstLabel);
disp("Accuracy on PCA+LDA reduced data = " + accuracy * 100 + "%");
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'), 'LDA on PCA reduced data');
