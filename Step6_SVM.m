%% SVM on LDA-only reduced data

% Clear workspace
clear
close all
clc

% Load LDA reduced data and labels
load('Reduced_data_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

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

%% SVM on PCA+LDA reduced data

% Clear workspace
clear
close all

% Load PCA+LDA reduced data and labels
load('Reduced_data_PCA_and_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Create an SVM template and specify polynomial kernel of order 5
tempSVM = templateSVM('KernelFunction','polynomial','PolynomialOrder',5);

% Fit the multiclass SVMs to the training data, using the template SVM
Mdl = fitcecoc(trn_proj,trnLabel,'Learners',tempSVM);

% Test the multiclass SVM model on the testing data
y_tst_predict = predict(Mdl,tst_proj);

% Check accuracy of predicted class labels
accuracy = mean(y_tst_predict==tstLabel) * 100;
disp("Accuracy with PCA + LDA reduced data: " + accuracy + "%");
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'),'SVM on PCA+LDA reduced data');

%% SVM on unreduced 324 features

% Clear workspace
clear
close all

% Load unreduced data and labels
load('CW2Data.mat','trnLabel','tstLabel','trn_features','tst_features');

% Create an SVM template and specify polynomial kernel of order 5
tempSVM = templateSVM('KernelFunction','gaussian');

% Fit the multiclass SVMs to the training data, using the template SVM
Mdl = fitcecoc(trn_features,trnLabel,'Learners',tempSVM,'FitPosterior',true);

% Test the multiclass SVM model on the testing data
y_tst_predict = predict(Mdl,tst_features);

% Check accuracy of predicted class labels
accuracy = mean(y_tst_predict==tstLabel) * 100;
disp("Accuracy with unreduced data: " + accuracy + "%");
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'),'SVM on unreduced data');

%% SVM on PCA-only reduced data
% For highest accuracy of 62.1% re-run the script in Step2 setting PCA to 
% reduce the data to 62 features. Once finished, please set it back to 55
% features which is optimal value for other classifiers.

% Clear workspace
clear
close all

% Load PCA reduced data and labels
load('Reduced_data_PCA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Create an SVM template and specify polynomial kernel of order 5
tempSVM = templateSVM('KernelFunction','polynomial','PolynomialOrder',5);

% Fit the multiclass SVMs to the training data, using the template SVM
Mdl = fitcecoc(trn_proj,trnLabel,'Learners',tempSVM);

% Test the multiclass SVM model on the testing data
y_tst_predict = predict(Mdl,tst_proj);

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict==tstLabel) * 100;
disp("Accuracy with PCA reduced data: " + accuracy + "%");
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'),'SVM on PCA reduced data');
