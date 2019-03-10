function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y); % number of training examples



% Nomalization Equation solution
% theta = pinv((X' * X))*X'*y;
fprintf('theta: %d\n',theta);

h = X*theta;
fprintf('h: %d\n',h);

errors = h - y;

fprintf('errors: %d\n',errors);

errors_sqr = errors.^2;

totalCost = sum(errors_sqr);

J = totalCost/(2*m);


end
