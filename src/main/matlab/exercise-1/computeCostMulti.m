function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

fprintf('theta: %d\n',theta);

h = X*theta;
fprintf('h: %d\n',h);

errors = h - y;

errors_sqr = errors.^2;

totalCost = sum(errors_sqr);

J = totalCost/(2*m);


end
