function [C, sigma] = dataset3Params(X, y, Xval, yval)
% Returns the choice of C and sigma
%  where you select the optimal (C, sigma) learning parameters to use for SVM
%  with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. (based on a cross-validation set)
%

% Return the optimal C and sigma learning parameters found using the cross validation set.
C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30];
sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30];
length_test = length(C_test);

for i=1:length_test
    for j=1:length_test
        C = C_test(i);
        sigma = sigma_test(j);
        
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        error(i,j)=mean(double(predictions ~= yval));
        
    end
end


min_mat=min(error(:));
[e_row,e_col]=find(error==min_mat);
C=C_test(e_row);
sigma=sigma_test(e_col);

end
