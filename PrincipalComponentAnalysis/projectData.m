function Z = projectData(X, U, K)
% Computes the reduced data representation when projecting only on to the top k eigenvectors
%   It returns the projected examples in Z.

% Set variables
Z = zeros(size(X, 1), K);

% The projection of the data using only the top K eigenvectors in U (first K columns). 

for i=1:size(X, 1)
    x = X(i, :)';
    for k=1:K
        Z(i,k)=x' * U(:, k);
    end
end

end
