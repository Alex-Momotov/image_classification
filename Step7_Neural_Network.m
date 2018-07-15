%% Neural Network using PCA+LDA reduced data

% Clear workspace
clear
close all
clc

% Load PCA+LDA reduced data and labels.
% Change 'PCA_and_LDA' to 'PCA' or 'LDA' to run with different data
% reduction methodology of our dimensionality reduction in Step 2.
load('Reduced_data_PCA_and_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Orientate data correctly for our NN
trn_proj = trn_proj';
tst_proj = tst_proj';
trnLabel = trnLabel';
tstLabel = tstLabel';

% Specify layer structure.
% When running the 'PCA' configuration change numFeatures to 55.
% When running 'PCA+LDA' or 'LDA' configuration change numFeatures to 9.
numClasses = 10;
numFeatures = 9;
layers = [
    sequenceInputLayer(numFeatures)  
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% Choose SGDM solver, 160 epochs, dynamically shrinking learning rate
options = trainingOptions(...
    'sgdm', ...
    'MaxEpochs', 160, ...
    'Plots','training-progress', ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod',50, ...
    'LearnRateDropFactor',0.1);

% Train our NN and use it to predict testing data
net = trainNetwork(trn_proj,categorical(trnLabel),layers,options);
y_tst_predict = classify(net,tst_proj);
y_tst_predict = grp2idx(y_tst_predict)';

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict==tstLabel) * 100;
disp("Accuracy with PCA+LDA reduced data: " + accuracy + "%");
plotconfusion(ind2vec(tstLabel),ind2vec(y_tst_predict),'Neural Network');
