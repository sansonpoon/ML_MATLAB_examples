function p = predict(Theta1, Theta2, X)
%Predict the label of an input given a trained neural network
%   Outputs the predicted label of X given the trained weights of 
%   a neural network (Theta1, Theta2)

% Initialize values
m = size(X, 1);

% add one to the column
X = [ones(m, 1) X];

% feedforward propagation algorithm
z2=X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];
z3=a2*Theta2';
a3 = sigmoid(z3);
[~,p]=max(a3, [], 2);

end
