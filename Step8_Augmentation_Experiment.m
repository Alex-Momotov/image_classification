%% Agumentation Experiment
% This part augments PCA+LDA feature space with handpicked PCA features
% with high class separability identified during Exploratory Data Analysis 

% Clear workspace
clear
close all
clc

% Load PCA reduced data
load('Reduced_data_PCA.mat');
trn_PCA = trn_proj;
tst_PCA = tst_proj;

% Load PCA+LDA reduced data and labels
load('Reduced_data_PCA_and_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Augment PCA+LDA data with some PCA features by adding right columns
featuresAdded = 5;
trn_proj(:,10:featuresAdded+9) = trn_PCA(:,1:featuresAdded);
tst_proj(:,10:featuresAdded+9) = tst_PCA(:,1:featuresAdded);

%% Kmeans clustering and classification with augmented data

% Fit LDA model and extract class centroid coordinates
ldaModel = fitcdiscr(trn_proj,trnLabel, 'DiscrimType','quadratic');
LDA_centroids = ldaModel.Mu;

% Set up kmeans parameters
k = 10;
maximumIter = 400;
startType = LDA_centroids;
nRep = 1;

% perform k-means clustering
[idx, C] = kmeans(trn_proj, k, 'Start', startType, ...
           'MaxIter',maximumIter, 'Replicates', nRep);

% Use k-means model to classify testing data
y_tst_predict = predict(ldaModel,tst_proj);

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict == tstLabel);
disp("k-means accuracy on PCA+LDA reduced data = " + accuracy*100 + "%");
figure;
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'), 'GMM on PCA reduced data');

%% GMM clustering and classification
% This part uses GMM in an unorthodox way to perform cassification. We
% intentionally prevent GMM from converging by setting the number of
% iterations to 1. We set initial data label guesses to training data.

% set up GMM parameters
k = 10;
startType = trnLabel;
options = statset('MaxIter',1);

% fit GMM to data and cluster original observations
GMM = fitgmdist(trn_proj,k,'Start',startType,'Options',options);
idx = cluster(GMM,trn_proj);

% Use trained GMM to classify testing data
y_tst_predict = cluster(GMM,tst_proj);

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict == tstLabel);
disp("GMM accuracy on PCA+LDA reduced data = " + accuracy*100 + "%");
figure;
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'), 'GMM on PCA+LDA reduced data');

%% SVM with augmented data

% Create an SVM template and specify polynomial kernel of order 5
tempSVM = templateSVM('KernelFunction','polynomial','PolynomialOrder',5);

% Fit the multiclass SVMs to the training data, using the template SVM
Mdl = fitcecoc(trn_proj,trnLabel,'Learners',tempSVM);

% Test the multiclass SVM model on the testing data
y_tst_predict = predict(Mdl,tst_proj);

% Check accuracy of predicted class labels
accuracy = mean(y_tst_predict==tstLabel) * 100;
disp("Accuracy with LDA-only reduced data: " + accuracy + "%");
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'),'SVM on LDA reduced data');

%% Neural Network with augmented data


% Orientate data correctly for our NN
trn_proj = trn_proj';
tst_proj = tst_proj';
trnLabel = trnLabel';
tstLabel = tstLabel';

% Specify layer structure.
% When running the 'PCA' configuration change numFeatures to 55.
% When running 'PCA+LDA' or 'LDA' configuration change numFeatures to 9.
numClasses = 10;
numFeatures = 14;
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
