function visualizeFit(X, mu, sigma2)
% Visualize the dataset and its estimated distribution.
%  Each example has a location (x1, x2) that depends on its feature values.
%

[X1,X2] = meshgrid(0:.5:35); 
Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
Z = reshape(Z,size(X1));

plot(X(:, 1), X(:, 2),'bx');
hold on;
colormap(winter)
% Do not plot if there are infinities
if (sum(isinf(Z)) == 0)
    contour(X1, X2, Z, 10.^(-20:2:0)','LineWidth',1.1);
end
hold off;

end