function [J, grad] = costFunctionReg(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization:
%   computes the cost of using theta as the parameter for regularized 
%   logistic regression and the gradient of the cost w.r.t. to the parameters.
% lambda is the parameter of the regularization

% Initialize some useful values
m = length(y); % number of training examples
grad = zeros(size(theta));
costsumterms = 0;
gradsumterms = zeros(1,size(X,2));

% Calculate the cost
for i=1:m
    hthetax = sigmoid(sum(theta'.*X(i,:)));
    costsumterms = costsumterms + ( -y(i,1)*log(hthetax) - (1-y(i,1))*log(1-hthetax) );
    thetasqsumterms=0;
    for j=1:size(X,2)
        gradsumterms(1,j) = gradsumterms(1,j) + (hthetax-y(i)).*X(i,j);
        if j>1
            thetasqsumterms = thetasqsumterms + (theta(j)*theta(j));
        end
        
    end
end
J=(costsumterms/m)+(lambda*thetasqsumterms/(2*m));

%Calculate the gradient
for j=1:size(X,2)
    if j==1
        grad(j)=gradsumterms(1,j)/m;
    else
        grad(j)=(gradsumterms(1,j)/m) + (lambda*theta(j)/m);
    end
end

end
