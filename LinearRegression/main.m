clc 
clearvars
close all

data = load('population_profit.txt'); % read in the data
x = data(:, 1); y = data(:, 2);

%Implementation
m = length(x); % number of training examples
X = [ones(m,1),data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters
iterations = 1500;
alpha = 0.01;

num_iters = iterations;
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Perform a single gradient step on the parameter vector  theta. 
    theta_old=theta;
    theta(1)=theta_old(1)-(alpha/m)*sum(theta_old(1)+theta_old(2)*X(:,2)-y(:));
    theta(2)=theta_old(2)-(alpha/m)*sum((theta_old(1)+theta_old(2)*X(:,2)-y(:)).*X(:,2));

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

fprintf('Theta computed from gradient descent:\n theta0=%f,\n theta1=%f\n',theta(1),theta(2))
% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 7] *theta;
fprintf('For population = 70,000, we predict a profit of $%f\n', predict1*10000);
predict2 = [1, 14] * theta;
fprintf('For population = 140,000, we predict a profit of $%f\n', predict2*10000);


% Plotting the training data with linear regression fit
figure(1)
plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s') % Set the y-axis label
xlabel('Population of City in 10,000s') % Set the x-axis label
% Plot the linear fit
hold on % keep previous plot visible
plot(X(:,2), X*theta, 'b-','LineWidth',1.5)
legend('Training data', 'Linear regression','Location','southeast')
hold off % don't overlay any more plots on this figure
xlim([4 25]) % set range of x-axis
set(gca,'LineWidth',2.0,'XScale','linear','FontSize',15)
print(gcf, '-r200','LRdata.png','-dpng')

% Visualizing J(theta)
% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
        t = [theta0_vals(i); theta1_vals(j)];
        J_vals(i,j)=(sum(((t(1)+t(2).*X(:,2))-y(:)).^2))/(2*m);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Plot the cost function J(theta)
figure(2)
subplot(1,2,1)
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0')
ylabel('\theta_1')
zlabel('J(\theta)')
shading interp
box on
set(gca,'LineWidth',1.5,'FontSize',12)
subplot(1,2,2)
% Plot the optimal point for theta_0 and theta_1 
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold on
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0')
ylabel('\theta_1')
hold off
legend({'Global mininum'})
set(gca,'LineWidth',1.5,'FontSize',12)
set(gcf,'Position',[0 0 800 360])
print(gcf, '-r200','CostFunction.png','-dpng')

disp('-----End-----')


%% Function
function J = computeCost(X, y, theta)
% Compute cost for linear regression:
%   computes the cost of using theta as the parameter for linear regression
%   to fit the data points in X and y

% Initialize values
m = length(y); % number of training examples

% Compute the cost of a particular choice of theta
J=(sum(((theta(1)+theta(2).*X(:,2))-y(:)).^2))/(2*m);
end