function centroids = computeCentroids(X, idx, K)
%Returns the new centroids by computing the means of the data points 
% assigned to each centroid.
%   It is given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. 

% Set variables
[m,n] = size(X);

centroids = zeros(K, n);

% Go over every centroid and compute mean of all points that belong to it. 

Ck=groupcounts(idx);
for k=1:K
	XSum = sum(X((idx==k),:));
    centroids(k,:)=(1/Ck(k))*XSum;
end

end

