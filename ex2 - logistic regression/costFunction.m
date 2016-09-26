function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hyp=sigmoid(X*theta);
tmp=zeros(m, 1);
%tmp0=zeros(m, 1)
tmpx=zeros(m, length(theta));
for i =1:m
    tmp(i)= -y(i) * log(hyp(i))-(1-y(i))*log(1-hyp(i));
    for j=1:length(theta);
   % tmp0(i)=hyp(i)-y(i)
    tmpx(i, j)=(hyp(i)-y(i)).*X(i, j);
    end
     
end
J=sum(tmp)*(1/m);

for j=1:length(theta)
    grad(j)=(1/m) * sum(tmpx(:, j));
end

%for
%grad(1)=(sum(tmp2)*(1/m)
%sum(tmp3)*(1/m), sum(tmp4)*(1/m))


% =============================================================

end
