function plotFit(min_x, max_x, mu, sigma, theta, p)
% Plots a learned polynomial regression fit over an existing figure.
%  Also works with linear regression.
%  Plots the learned polynomial fit with power p and feature normalization 
%  (mu, sigma).

% Hold on to the current figure
hold on;

% Plot a range slightly bigger than the min and max values to get
% an idea of how the fit will vary outside the range of the data points
x = (min_x - 15: 0.05 : max_x + 25)';

% Map the X values 
X_poly = polyFeatures(x, p);
X_poly = bsxfun(@minus, X_poly, mu);
X_poly = bsxfun(@rdivide, X_poly, sigma);

% Add ones
X_poly = [ones(size(x, 1), 1) X_poly];

% Plot
plot(x, X_poly * theta, '--', 'LineWidth', 2)

% Hold off to the current figure
hold off

end
