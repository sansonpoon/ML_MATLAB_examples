function centroids = kMeansInitCentroids(X, K)
%This function initializes K centroids that are to be used in K-Means on the dataset X

% Set centroids to randomly chosen examples from the dataset X
% Initialize the centroids to be random examples
% Randomly reorder the indicies of examples
randidx = randperm(size(X,1));
% Take the first K examples
centroids = X(randidx(1:K),:);

end

