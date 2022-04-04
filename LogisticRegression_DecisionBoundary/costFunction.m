function [J, grad] = costFunction(theta, X, y)
% Compute cost and gradient for logistic regression:
%   computes the cost of using theta as the parameter for logistic 
%   regression and the gradient of the cost w.r.t. to the parameters.
% Note: grad have the same dimensions as theta

% Initialize values
m = length(y); % number of training examples
costsumterms = 0;
gradsumterms = zeros(1,size(X,2));

for i=1:m
    hthetax = sigmoid(sum(theta'.*X(i,:)));
    costsumterms = costsumterms + ( -y(i,1)*log(hthetax) - (1-y(i,1))*log(1-hthetax) );
    for j=1:size(X,2)
        gradsumterms(1,j) = gradsumterms(1,j) + (hthetax-y(i)).*X(i,j);
    end
end

J=costsumterms/m; % cost
grad=gradsumterms./m; % gradient
end
