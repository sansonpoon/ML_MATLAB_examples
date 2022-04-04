function X_rec = recoverData(Z, U, K)
% Recovers an approximation of the original data when using the projected data
%   It returns the approximate reconstruction in X_rec.
%

% Set variables.
X_rec = zeros(size(Z, 1), size(U, 1));

% Compute the approximation of the data by projecting back
%  onto the original space using the top K eigenvectors in U.
%
% Notice that U(j, 1:K) is a row vector.      
for i=1:size(Z, 1)
    v = Z(i, :)';
    for j=1:size(U, 1)
        X_rec(i,j) = v' * U(j, 1:K)';
    end
end
    
end
