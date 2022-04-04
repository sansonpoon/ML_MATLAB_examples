function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
% Compute cost and gradient for regularized linear regression with multiple 
%  variables
% Using theta as the parameter for linear regression to fit the data
%  points in X and y. Returns the cost in J and the gradient in grad

% Initialize values
m = length(y); % number of training examples

% Compute the cost 
h=X*theta;
J = (1/(2*m))*(sum((h-y).^2)) + ((lambda/(2*m))*sum(theta(2:end).^2));

% Compute the gradient
thetaZero = theta;
thetaZero(1) = 0;
grad = ((1/m)*sum((h-y).*X)) + ((lambda/m) * thetaZero');







% =========================================================================

grad = grad(:);

end
