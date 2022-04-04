function sim = gaussianKernel(x1, x2, sigma)
% Returns a radial basis function kernel between x1 and x2

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% Return the similarity between x1 and x2 computed using a Gaussian kernel 
%  with bandwidth sigma.
sim=exp(-sum((x1-x2).^2)/(2.*sigma.*sigma));
    
end
