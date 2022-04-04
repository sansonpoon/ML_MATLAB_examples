function W = debugInitializeWeights(fan_out, fan_in)
% Initializes the weights of a layer with fan_in incoming connections and 
%  fan_out outgoing connections using a fix set of values

% Set W to zeros
W = zeros(fan_out, 1 + fan_in); % the first row of W handles the "bias" terms

% Initialize W using "sin", this ensures that W is always of the same
% values and will be useful for debugging
W = reshape(sin(1:numel(W)), size(W)) / 10;

end
