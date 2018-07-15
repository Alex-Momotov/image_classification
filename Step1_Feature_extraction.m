% This script must be in the same folder as 'CW2Data.mat' because we append
% the computed features to that file

% Clear workspace
clear
close all
clc

% Load image data
load('CW2Data.mat');

% Create feature matrix for training data
trn_features = nan(10000,324);
for i = 1 : 10000
    trn_features(i,:) = ComputeFeature(trnImage(:,:,:,i));
end

% Create feature matrix for testing data
tst_features = nan(1000,324);
for i=1 : 1000
    tst_features(i,:) = ComputeFeature(tstImage(:,:,:,i));
end

% Append computed feature matrices to our data file
save('CW2Data.mat','trn_features','tst_features','-append');