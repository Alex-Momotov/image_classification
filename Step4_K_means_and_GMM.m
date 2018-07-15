%% K-means clustering, initialise centroid seeds with LDA
% This section performs K-means clustering, but instead of choosing
% cluster centroids randomly, we assign them using centroid coordinates
% obtained from LDA model. This enables us to perform classification.

% Load PCA+LDA reduced data and labels.
% Change 'PCA_and_LDA' to 'PCA' or 'LDA' to run with different data
% reduction methodology of our dimensionality reduction in Step 2.
clear; close all;
load('Reduced_data_PCA_and_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% Fit LDA model and extract class centroid coordinates
ldaModel = fitcdiscr(trn_proj,trnLabel,'DiscrimType','quadratic');
LDA_centroids = ldaModel.Mu;

% Set up kmeans parameters
k = 10;
maximumIter = 400;
startType = LDA_centroids;
nRep = 1;

% perform k-means clustering
[idx, C] = kmeans(trn_proj, k, 'Start', startType, ...
           'MaxIter',maximumIter, 'Replicates', nRep);

% Visualise clusters and cluster centres from K-means results
figure
hold on
gscatter(trn_proj(:,1),trn_proj(:,2),idx)
plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',2)
title("K-means clustering");
hold off

figure; hold on;
scatter3(trn_proj(:,1),trn_proj(:,2),trn_proj(:,3),6,idx,'.')
scatter3(C(:,1),C(:,2),C(:,3),100,'filled','black')
title("K-means clustering 3D");

% Check resulting performance by plotting truth labels
figure
gscatter(trn_proj(:,1),trn_proj(:,2),trnLabel)
title("Truth labelled classes for comparison");

figure
scatter3(trn_proj(:,1),trn_proj(:,2),trn_proj(:,3),15,trnLabel,'.')
title("Truth labelled classes for comparison");

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

% Load PCA+LDA reduced data and labels.
% Change 'PCA_and_LDA' to 'PCA' or 'LDA' to run with different data
% reduction methodology of our dimensionality reduction in Step 2.
clear; close all;
load('Reduced_data_PCA_and_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

% set up GMM parameters
k = 10;
startType = trnLabel;
options = statset('MaxIter',1);

% fit GMM to data and cluster original observations
GMM = fitgmdist(trn_proj,k,'Start',startType,'Options',options);
idx = cluster(GMM,trn_proj);

% Plot GMM clusters
figure; hold on;
gscatter(trn_proj(:,1),trn_proj(:,2),idx);
plot(GMM.mu(:,1),GMM.mu(:,2),'kx','LineWidth',2,'MarkerSize',10)
title 'GMM Clustering'
xlabel('D1')
ylabel('D2')

% Check resulting performance by plotting truth labels
figure
gscatter(trn_proj(:,1),trn_proj(:,2),trnLabel)
title("Truth labelled classes for comparison");

% Use trained GMM to classify testing data
y_tst_predict = cluster(GMM,tst_proj);

% Check accuracy of predicted values and display confusion matrix
accuracy = mean(y_tst_predict == tstLabel);
disp("GMM accuracy on PCA+LDA reduced data = " + accuracy*100 + "%");
figure;
plotconfusion(ind2vec(tstLabel'),ind2vec(y_tst_predict'), 'GMM on PCA+LDA reduced data');

