function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


for iter = 1:num_iters
    h = X*theta;
    errors = h - y;
    gradient = X'*errors;
    theta_change = gradient;

    for i = 1 : length(gradient)
        theta_change(i) = gradient(i) * alpha/m;
    end

    theta = theta - theta_change;
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
