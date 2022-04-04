clc
clear
file = 'emailSample1.txt';
contents = fileread(file);

% Convert contents to a string
processed = string(contents);

% Lower-case the string using lower.
processed= lower(processed);
% Strip HTML tags using eraseBetween
processed= eraseBetween(processed,'<','>','Boundaries','inclusive');
% Replace dollar signs with 'dollar'
processed = replace(processed,"$","dollar");
% Split the email on whitespace into individual tokens (contents will become a string array)
processed = split(processed);
% Normalize URLs using contains and logical indexing
processed(startsWith(processed,["http://","https://"])) = "httpaddr";
% Normalize email addresses and numbers, and remove punctuation using regexprep.
processed = regexprep(processed,'.+@.+',"emailaddr");
processed = regexprep(processed,'[0-9]+',"number");
processed = regexprep(processed,'[^a-zA-Z0-9]',"");
% Remove empty or single character words using strlength
processed(strlength(processed)<=1) = [];

% Stem the words in the string array
processed = arrayfun(@(str)string(porterStemmer(char(str))),processed);

% Import the vocabulary list
vocab = readtable('vocab.txt');
vocab = string(vocab{:,2});

% Map the string array values to the vocabulary list
features = double(ismember(vocab,processed));
table(vocab,features)

% Train an SVM for spam classification using fitclinear
load spamTrain.mat;
opts = struct('Holdout',0.3);
linSVMmdl = fitclinear(X,y,'Learner','svm','Regularization','ridge','OptimizeHyperparameters',{'lambda'},'HyperparameterOptimizationOptions',opts);
load('spamTest.mat');
fprintf('Training Accuracy: %f | Test Accuracy: %f\n',...
        mean(predict(linSVMmdl,X)==y)*100,...
        mean(predict(linSVMmdl,Xtest)==ytest)*100);

% List the top predictors for spam    
% Sort the weights and obtain the vocabulary list
tbl = table(vocab,linSVMmdl.Beta,'VariableNames',{'Word','Score'});
tbl = sortrows(tbl,'Score','descend');
%%
yourEmail = '';
eml = 'spamSample2.txt';
processed = preprocess(eml);
features = ismember(vocab,processed);
if predict(linSVMmdl,double(features)')
disp('This email is probably spam.')
else
disp('This email is probably not spam.')
end

%--------------------------------------------------------------------------
% Functions

% preprocess
function contents = preprocess(file)
contents = string(fileread(file));
contents = lower(contents);
contents = eraseBetween(contents,'<','>','Boundaries','inclusive');
contents = replace(contents,"$","dollar ");
contents = split(contents);
contents(startsWith(contents,["http://","https://"])) = "httpaddr";
contents = regexprep(contents,'.+@.+',"emailaddr");
contents = regexprep(contents,'[0-9]+',"number");
contents = regexprep(contents,'[^a-zA-Z0-9]',"");
contents(strlength(contents)<=1) = [];
contents = arrayfun(@(str)string(porterStemmer(char(str))),contents);
end

% decisionBoundary
function decisionBoundary(mdl,X,y)
    figure; hold on;
    x1 = linspace(min(X(:,1)),max(X(:,1)));
    x2 = linspace(min(X(:,2)),max(X(:,2)));
    [Xgrid,Ygrid] = meshgrid(x1,x2);
    Z = reshape(predict(mdl,[Xgrid(:),Ygrid(:)]), size(Xgrid));
    contour(Xgrid,Ygrid,Z,'Levels',1); 
    plot(X(y==1,1),X(y==1,2),'k+','MarkerSize',7);
    plot(X(~y,1),X(~y,2),'ko','MarkerFaceColor','y','MarkerSize',7); 
    misclassidx = y~=predict(mdl,X);
    plot(X(misclassidx,1),X(misclassidx,2),'rx')
    hold off;
end








