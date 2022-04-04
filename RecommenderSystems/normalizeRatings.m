function [Ynorm, Ymean] = normalizeRatings(Y, R)
% Preprocess data by subtracting mean rating for every movie (every row)
%   Normalized Y so that each movie has a rating of 0 on average, 
%   Returns the mean rating in Ymean.
%

[m, n] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    idx = find(R(i, :) == 1);
    Ymean(i) = mean(Y(i, idx));
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
end

end
