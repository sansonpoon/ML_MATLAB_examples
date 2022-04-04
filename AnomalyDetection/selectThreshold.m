function [bestEpsilon,bestF1] = selectThreshold(yval, pval)
% Find the best threshold (epsilon) to use for selecting outliers.

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    % Compute the F1 score of choosing epsilon as the threshold and place the value in F1. 
    predictions = (pval < epsilon);
    % outliers = 1
    tp = sum((yval==1) & (predictions==1)); % true positives
    fp = sum((yval==0) & (predictions==1)); % false positives
    fn = sum((yval==1) & (predictions==0)); % false negatives
    
    prec = tp/(tp+fp); % precision
    rec  = tp/(tp+fn); % recall
    % F1 score
    F1 = (2*prec*rec)/(prec+rec); 
    
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
