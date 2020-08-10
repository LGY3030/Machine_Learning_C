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
%
find_C = [0.01;0.03;0.1;0.3;1;3;10;30];
find_sigma = [0.01;0.03;0.1;0.3;1;3;10;30];
final_C = 0;
final_sigma = 0;
predict = 100;
for i = 1:8,
	for j = 1:8,
		model= svmTrain(X, y, find_C(i), @(x1, x2) gaussianKernel(x1, x2, find_sigma(j))); 
		pred = svmPredict(model, Xval);
		get = mean(double(pred ~= yval));
		if get<predict,
			final_C = find_C(i);
			final_sigma = find_sigma(j);
			predict = get;
		end;
	end;
end;

C = final_C;
sigma = final_sigma;



% =========================================================================

end
