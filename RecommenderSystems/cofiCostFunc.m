function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
% Collaborative filtering cost function
%   Returns the cost and gradient for the collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% Compute the cost function and gradient for collaborative filtering. 
%  1) Implement the cost function (without regularization)
%  2) Implement the gradient and use the checkCostFunction routine to check
%     that the gradient is correct.
%  3) Implement regularization.
%
%        X          - num_movies x num_features matrix of movie features
%        Theta      - num_users x num_features matrix of user features
%        Y          - num_movies x num_users matrix of user ratings of movies
%        R          - num_movies x num_users matrix, where R(i, j) = 1 if the 
%                     i-th movie was rated by the j-th user
%        X_grad     - num_movies x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

ThetaTx = X*Theta';
J=0.5*sum(sum((R.*ThetaTx-Y).^2));

X_grad = (R.*ThetaTx-Y) *Theta;
Theta_grad = (R.*ThetaTx-Y)' *X;

J = J + ((lambda/2)*sum(sum(Theta.^2))) + ((lambda/2)*sum(sum(X.^2)));
    
X_grad = X_grad + lambda*X;
Theta_grad = Theta_grad + lambda*Theta;

grad = [X_grad(:); Theta_grad(:)];

end
