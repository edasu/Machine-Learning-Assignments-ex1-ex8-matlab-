function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
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

Cval = [0.01 0.03 0.1 0.3 1 3 10 30]';
Sval = Cval;

prediction_error = zeros(8, 8);
result = zeros(16,3);
eror = 1;

 for i = 1:length(Cval) %C
      for j = 1: length(Sval) %Sigma
          model = svmTrain(X, y, Cval(i), @(x1, x2) gaussianKernel(x1, x2, Sval(j)));
          predictions = svmPredict(model, Xval);
          prediction_error(i,j) = mean(double(predictions ~= yval));
          result(eror,:) = [prediction_error(i,j), Cval(i), Sval(j)]; %put all in result matrix
          eror = eror + 1;
      end
 end

sorted_result = sortrows(result, 1); % sort result

C = sorted_result(1,2); %choose C
sigma = sorted_result(1,3); %choose sigma
end
end
