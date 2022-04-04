function plotData(X, y)
% Plots the data points X and y into a new figure 

% Create New Figure
figure; hold on;

% 'k+' for the positive
% 'ko' for the negative examples.

% Find Indices of Positive and Negative Examples
pos = find(y == 1);
neg = find(y == 0);

% Plot 
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'LineWidth', 1,'MarkerSize', 7);







% =========================================================================



hold off;

end
