function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Cvec = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmavec = Cvec;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return thce optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Allocating memory
predictionMatrix = zeros(length(Cvec));

for i = Cvec
  for j = sigmavec
    model= svmTrain(X, y, i, @(X, y) gaussianKernel(X, y, j));  
    
    predictions = svmPredict(model, Xval);

    m = find(Cvec==i);
    n = find(sigmavec==j);
    
    predictionMatrix(m,n) = mean(double(predictions ~= yval));
  end
end

[ix,iy] = find(predictionMatrix == min(min(predictionMatrix)));

C = Cvec(ix);
sigma = sigmavec(iy);

% =========================================================================

end
