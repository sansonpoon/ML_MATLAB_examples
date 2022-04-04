function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
% Generate the train and validation errors needed to plot a validation 
%  curve that we can use to select lambda
% Returns the train and validation errors (in error_train, error_val)
%  for different values of lambda. You are given the training set (X,y) and
%  validation set (Xval, yval).

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% Initialize values
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% error_train(i), and error_val(i) give the errors obtained after training 
% with lambda = lambda_vec(i)

for i=1:length(lambda_vec)
    lambda=lambda_vec(i);
    theta = trainLinearReg(X, y, lambda);
    error_train(i,1)=linearRegCostFunction(X, y, theta, 0);
    error_val(i,1)=linearRegCostFunction(Xval, yval, theta, 0);
end

end
