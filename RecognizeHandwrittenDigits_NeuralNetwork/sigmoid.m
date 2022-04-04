function g = sigmoid(z)
% Compute sigmoid functoon
g = 1.0 ./ (1.0 + exp(-z));
end
