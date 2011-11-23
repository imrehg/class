function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% % Initial setup
% bestpred = 0

% % number of different parameter values to try each, this altogether
% % there will be nval^2 training
% nval = 15;
% Clist = logspace(log10(0.01), log10(30), nval);
% sigmalist = logspace(log10(0.01), log10(30), nval);

% for i = 1:nval
%   Ct = Clist(i);
%   for j = 1:nval
%    sigmat = sigmalist(j);
%    model = svmTrain(X, y, Ct, @(x1, x2) gaussianKernel(x1, x2, sigmat));
%    predictions = svmPredict(model, Xval);
%    correct = mean(double(predictions == yval));
%    if (correct > bestpred)  % if we are better than before, keep parameters
%      C = Ct;
%      sigma = sigmat;
%      bestpred = correct;
%    end
%  end
% end

% %Final results
% C
% sigma
% bestpred

C =  0.97035
sigma =  0.098506


% =========================================================================

end
