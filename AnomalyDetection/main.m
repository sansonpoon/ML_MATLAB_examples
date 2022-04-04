clc
clearvars
clear
close all


% The following command loads the dataset. You should now have the variables X, Xval, yval in your environment
load('data1.mat');

% Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');


%  Estimate mu and sigma2
[mu, sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Visualize the fit
figure(1)
visualizeFit(X,  mu, sigma2)
hold on
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
xlim([0 30])
ylim([0 30])
%  Draw a red circle around those outliers
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 1.5, 'MarkerSize', 10)
hold off
set(gca,'LineWidth',1.5,'XScale','linear','FontSize',15)
print(gcf, '-r200','ClassifiedAnomalies.png','-dpng')

