function [X_poly] = polyFeatures(X, p)
% Maps X (1D vector) into the p-th power.
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];

X_poly = zeros(numel(X), p);

% Return a matrix X_poly where the p-th column of X contains the values of 
%  X to the p-th power.
for j=1:p
    X_poly(:,j)=X.^j;
end

end
