function g = sigmoidGradient(z)
% Returns the gradient of the sigmoid function evaluated at z
g = sigmoid(z).*(1-sigmoid(z));

end
