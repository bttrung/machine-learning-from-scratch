function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

totalTheta = 0;
for i = 2:length(theta)
    totalTheta = totalTheta + theta(i)*theta(i);
end

h = sigmoid(X*theta);
J = (1/m) * (-y'*log(h) - (1 - y)'*log(1-h)) + (lambda / (2*m)) * totalTheta;

% You need to return the following variables correctly
errors = h - y;

grad = (X' * errors)/m;


for i = 2:length(grad)
    grad(i) = grad(i) + (lambda * theta(i)) / m;
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
