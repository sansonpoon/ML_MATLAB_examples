function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Implements the neural network cost function for a two layer neural network
%  which performs classification
% Computes the cost and gradient of the neural network. The parameters
%  for the neural network are "unrolled" into the vector nn_params
%  and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad is a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup variables
m = size(X, 1);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%==========================================================================
% Part 1-------------------------------------------------------------------
% Feedforward the neural network and return the cost in the variable J.

KClasses = eye(10);
yki = KClasses(y,:);

a1=[ones(m, 1) X];
z2=a1*Theta1';
a2=sigmoid(z2);
a2=[ones(size(a2,1), 1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
h=a3;

for k=1:num_labels
    J=J+sum((-yki(:,k).*log(h(:,k))) - ((1-yki(:,k)).*(log(1-h(:,k)))));
end
J=J/m;
% end of Feedforward and Cost Function (30)
Theta1r=Theta1(:,2:end);
Theta2r=Theta2(:,2:end);

reg_term = (lambda/(2*m))*(sum(sum(Theta1r.*Theta1r))+sum(sum(Theta2r.*Theta2r)));

J=J+reg_term;

%==========================================================================
% Part 2-------------------------------------------------------------------
% Implement the backpropagation algorithm to compute the gradients
%  Theta1_grad and Theta2_grad.

for t=1:m
    for k=1:num_labels
        yk=yki(t,k);
        delta3(k)=a3(t,k)-yk;
    end
    delta2=(Theta2'*delta3').*sigmoidGradient([1, z2(t,:)])';
    delta2r=delta2(2:end);
    Theta2_grad = Theta2_grad+(delta3'*a2(t,:));
    Theta1_grad = Theta1_grad+(delta2r*a1(t,:));
end
Theta1_grad=Theta1_grad/m;
Theta2_grad=Theta2_grad/m;

%==========================================================================
% Part 3-------------------------------------------------------------------
%Implement regularization with the cost function and gradients.

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1r;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2r;

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
