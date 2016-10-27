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

Cs=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
sigmas=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30]
para= zeros(64, 2)

k=1
for j=1:8
    for i=1:8
        para(k, :)=[Cs(i), sigmas(j)]
        k=k+1
    end
end

for i=1:64
    model(i)= svmTrain(X, y, para(i, 1), @(x1, x2) gaussianKernel(x1, x2, para(i, 2)));
end

for i=1:64
    predictions(:,i) = svmPredict(model(i), Xval);
end

for i=1:64
    errors(:, i)= mean(double(predictions(:, i) ~= yval));
end

good_para=find(errors == min(errors))
C=para(good_para, 1)
sigma=para(good_para, 2)


% =========================================================================

end
