function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y);
J = 0;
hyp=X*theta;
tmp=zeros(m, 1)

for i=1:m
    tmp(i)=(hyp(i)-y(i)).^2;
end;
J=sum(tmp)*1/(2*m)
end
