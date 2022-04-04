function [J, grad] = lrCostFunction(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization:
%   computes the cost of using theta as the parameter for regularized
%    logistic regression and the gradient of the cost w.r.t. to the 
%    parameters. 

% Initialize values
m = length(y); % number of training examples

% Compute the cost J
XTheta = X * theta;
h_Theta = sigmoid(XTheta);
sum_calJ = (-y'*log(h_Theta))-((1-y')*log(1-h_Theta));
lambdaterm = (lambda/(2*m))*sum(theta(2:length(theta)).^2);
J=((1/m)*sum(sum_calJ))+lambdaterm;

% Compute the partial derivatives and set grad to the partial derivatives 
%  of the cost w.r.t. each parameter in theta
beta = h_Theta - y;
beta = repmat(beta, 1, size(X,2));
sum_calgrad = X.*beta;
grad=(1/m)*sum(sum_calgrad);
grad(:,2:length(grad))=grad(:,2:length(grad)) + (lambda/m)*theta(2:length(theta))';
grad = grad(:);

end
