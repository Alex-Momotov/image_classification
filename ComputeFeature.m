function Feature = ComputeFeature(Image)
%% Detail of ComputeHOG function
%  Compute the HOG feature for given Image with a cell size of [8,8], and
%  a block size of [2,2]
%  Image -- 3D image (RGB) dataset matrix (32 x 32 x 3)
%  Feature -- 1D HOG feature vector (1 x 324) Default

Feature = extractHOGFeatures(Image);

end

