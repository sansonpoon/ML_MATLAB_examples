function [all_theta] = oneVsAll(X, y, num_labels, lambda)
% Trains multiple logistic regression classifiers and returns all
%  the classifiers in a matrix all_theta, where the i-th row of all_theta 
%  corresponds to the classifier for label i

% Initialize values
m = size(X, 1);
n = size(X, 2);

all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Train num_labels logistic regression classifiers with regularization parameter lambda. 
for i=1:num_labels
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
[theta] = fmincg (@(t)(lrCostFunction(t, X, (y == i), lambda)), initial_theta, options);
all_theta(i,:)=theta(:);
end

end
