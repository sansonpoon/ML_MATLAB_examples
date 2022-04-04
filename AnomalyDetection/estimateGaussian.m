function [mu,sigma2] = estimateGaussian(X)
% This function estimates the parameters of a Gaussian distribution using the data in X
%   X is the dataset with each n-dimensional data point in one row
%   mu is an n-dimensional vector
%   sigma^2 is an n x 1 vector

% Set variables
[m, n] = size(X);

% mu(i) contain the mean of the data for the i-th feature;
% sigma2(i) contain variance of the i-th feature.
mu=sum(X)/m;
sigma2 = sum((X-mu).^2)/m;

end
