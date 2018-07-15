%% Clear workspace, load PCA+LDA reduced data

% Clear workspace
clear
close all
clc

% Load PCA+LDA reduced data and labels
load('Reduced_data_PCA_and_LDA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

%% Visualise our PCA+LDA reduced feature space

% Visualise entire feature space using all 10 classes. Rows and columns of
% scatter matrix represent dimensions, while colour groups represent classes
figure;
color = colormap(jet(10));
markersymbol = [];
markersize = [1.7];
drawLegend = [];
diagStyle = [];
featureLabels = {'D1'; 'D2'; 'D3'; 'D4';'D5';'D6';'D7';'D8';'D9'};
gplotmatrix(trn_proj,trn_proj,trnLabel,color,markersymbol,markersize,drawLegend,diagStyle,featureLabels,featureLabels);
title('Feature space scatter matrix for all 10 classes')

% Visualise dimensions 1 and 2
color = colorcube(10);
figure
gscatter(trn_proj(:,[1]),trn_proj(:,[2]),trnLabel,color,'.',5);
title('Dimensions 1 and 2 for all 10 classes')

% Visualise dimensions 1 and 3
figure
gscatter(trn_proj(:,[1]),trn_proj(:,[3]),trnLabel,color,'.',5);
title('Dimensions 1 and 3 for all 10 classes')

% Visualise dimensions 1, 2 and 3
figure
scatter3(trn_proj(:,1),trn_proj(:,2),trn_proj(:,3),15,trnLabel,'.')
title("3-feature scatter plot for all 10 classes");

%% Visualise each class individually
% This part individually highlights each class withing the feature space

labelNames = ["airplane","automobile","bird","cat","deer", ...
              "dog","frog","horse","ship","truck"];

for class = 1 : 10
    figure;
    color = [];
    markersymbol = [];
    markersize = [1.8];
    drawLegend = [];
    diagStyle = [];
    featureLabels = {'D1'; 'D2'; 'D3'; 'D4'; 'D5'};
    gplotmatrix(trn_proj(:,1:5),trn_proj(:,1:5),trnLabel==class,color,markersymbol,markersize,drawLegend,diagStyle,featureLabels,featureLabels);
    title("5-features scatter matrix for class: " + labelNames(class));
end

%% Visualise each class individually in 3D
labelNames = ["airplane","automobile","bird","cat","deer", ...
              "dog","frog","horse","ship","truck"];

for class = 1 : 10
    figure;
    scatter3(trn_proj(:,1),trn_proj(:,2),trn_proj(:,3),15,trnLabel==class,'.')
    title("3-feature scatter for class: " + labelNames(class));
end

%% Visualise class 'frog' against every other class

labelNames = ["airplane","automobile","bird","cat","deer", ...
              "dog","frog","horse","ship","truck"];
class1 = 7;
for dimension = 1 : 10
    % Select data for the class to plot
    class2 = dimension;
    logIdx = trnLabel == class1 | trnLabel == class2;
    labels = trnLabel(logIdx,:);
    data = trn_proj(logIdx,:);
    
    figure;
    scatter3(data(:,[1]),data(:,[2]),data(:,[3]),15,labels,'.');
    title('Scatter plot of classes: ' + labelNames(class1) + " vs " + labelNames(class2));
end

%% Explore class separability across all 9 dimensions using histograms

for dimension = 1 : 9
    figure;
    hold on;
    for class = 1 : 10
        logIdx = trnLabel == class;
        data = trn_proj(logIdx,dimension);
        histogram(data,40,'EdgeAlpha',0.0);
    end
    title("Dimension: " + dimension);
end

%% Explore class separability using Probability Density Function (PDF)
col = hsv(10);
for dimension = 1 : 9
    figure;
    hold on;
    for class = 1 : 10
        logIdx = trnLabel == class;
        data = trn_proj(logIdx,dimension);
        h = histfit(data,5,'kernel');
        h(1).EdgeAlpha = 0;
        h(1).FaceColor = 'none';
        h(2).Color = col(class,:);
    end
    title("Dimension: " + dimension);
end

%% Explore class separability using PCA-only reduced data and PDF
clc; clear; close all

% Load PCA reduced data and labels
load('Reduced_data_PCA.mat');
load('CW2Data.mat','trnLabel','tstLabel');

col = hsv(10);
for dimension = 1 : 20
    figure;
    hold on;
    for class = 1 : 10
        logIdx = trnLabel == class;
        data = trn_proj(logIdx,dimension);
        h = histfit(data,5,'kernel');
        h(1).EdgeAlpha = 0;
        h(1).FaceColor = 'none';
        h(2).Color = col(class,:);
    end
    title("Dimension: " + dimension);
end

%% Explore class separability of the unreduced data
% This section visualises some of the 324 unreduced features to see if we 
% can handpick dimensions with good class separability for classification

% Load raw 324 computed features and class labels
load('CW2Data.mat','trn_features','tst_features','trnLabel','tstLabel');

% Loops through dimensions of unreduced 324 features, change loop
% iterations '40 : 60' to explore other features

for dimension = 40 : 60
    color = colorcube(10);
    figure
    gscatter(trn_features(:,[dimension]),trn_features(:,[dimension+1]),trnLabel,color,'.',5);
    title("Dimensions " + dimension + " vs " + (dimension+1));
end
