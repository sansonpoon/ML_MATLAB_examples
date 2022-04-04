clc
clearvars
close all
clear

%--------------------------------------------------------------------------
% Load from data1:
% You will have X, y in your environment
load('data1.mat');

C = 1;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);

visualizeBoundaryLinear(X, y, model);
box on
set(gca,'LineWidth',1.5,'FontSize',15)
print(gcf, '-r200','SVM_data1.png','-dpng')

x1 = [1 2 1]; x2 = [0 4 -1]; 
sigma =2;
sim = gaussianKernel(x1, x2, sigma);
fprintf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f : \n\t%g\n', sigma, sim);

%--------------------------------------------------------------------------
% Load from data2: 
% You will have X, y in your environment
load('data2.mat');  

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run faster. However, in practice, 
% you will want to run the training to convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);
box on
set(gca,'LineWidth',1.5,'FontSize',15)
print(gcf, '-r200','SVM_data2.png','-dpng')

%--------------------------------------------------------------------------
% Load from data3: 
% You will have X, y in your environment
load('data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model = svmTrain(X, y, C, @(x1, x2)gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
box on
set(gca,'LineWidth',1.5,'FontSize',15)
print(gcf, '-r200','SVM_data3.png','-dpng')

disp('-----DONE-----')





