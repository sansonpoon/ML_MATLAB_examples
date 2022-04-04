function x = emailFeatures(word_indices)
% Takes in a word_indices vector and produces a feature vector from the word indices

% Total number of words in the dictionary
n = 1899;

x = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Return a feature vector for the given email (word_indices). 
% To help make it easier to process the emails, it has already pre-processed 
%  each email and converted each word in the email into an index in
%  a fixed dictionary (of 1899 words). 
% The variable word_indices contains the list of indices of the words
%  which occur in one email.
% 
%       Concretely, if an email has the text:
% 
%          The quick brown fox jumped over the lazy dog.
% 
%       Then, the word_indices vector for this text might look like:
% 
%           60  100   33   44   10     53  60  58   5
% 
%       where, we have mapped each word onto a number, for example:
% 
%           the   -- 60
%           quick -- 100
%           ...
% 
% (note: the above numbers are just an example and are not the actual mappings).
%
    
x(word_indices) = 1;

end
