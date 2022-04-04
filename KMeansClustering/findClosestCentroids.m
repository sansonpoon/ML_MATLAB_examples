function idx = findClosestCentroids(X, centroids)
% Computes the centroid memberships for every example.
%   Returns the closest centroids in idx for a dataset X where each row is 
%    a single example. 
%   idx = m x 1 vector of centroid assignments (i.e. each entry in range [1..K])

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

% Go over every example, find its closest centroid, and store the index 
%  inside idx at the appropriate location.

vecDist=zeros(K,1);
for i=1:size(X,1)
    for j=1:K
        vecDist(j,1)=(norm(X(i,:)-centroids(j,:)))^2;
    end
    [~, idx(i,1)]=min(vecDist);
end

end

