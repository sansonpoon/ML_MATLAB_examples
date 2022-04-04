function p = predict(theta, X)
% Predict whether the label is 0 or 1 using learned logistic regression
% parameters theta:
%   computes the predictions for X using a threshold at 0.5
%   (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples
p = zeros(m, 1);
for i=1:m
    hthetax = sigmoid(sum(theta'.*X(i,:)));
    if hthetax >=0.5
        p(i,1)=1;
    else
        p(i,1)=0;
    end
end

end
