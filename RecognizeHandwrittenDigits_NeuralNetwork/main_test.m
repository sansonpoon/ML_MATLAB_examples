clc
clearvars
close all
clear
%dir

load('MNIST_subset.mat');
% Test set (10% of data)
test_index = [1:10:size(X, 1)];
X=X(test_index,:);
y=y(test_index,1);

m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));
% Load saved matrices from file
%load('pretrainedweights.mat');
load('trainedweights.mat'); % using the weight trained in main_train.m
% Theta1 has size 25 x 401
% Theta2 has size 10 x 26

pred = predict(Theta1, Theta2, X);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y)) * 100);

% Randomly permute examples
rp = randi(m);
% Predict
pred = predict(Theta1, Theta2, X(rp,:));
fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
fprintf('It is actually a %d\n',mod(y(rp),10));
if pred==y(rp)
    disp('Correct!')
else
    disp('Not correct prediction. :(')
end
% Display
displayData(X(rp, :));

disp('---DONE---')
