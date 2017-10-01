function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
        
        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %
        % ============================================================

    for iter = 1:num_iters

        % calculate the hypotheses and the errors
        hypotheses = X * theta;
        errors = hypotheses - y;
        
        % Work out the "gradient" 
        gradients = X' * errors;

        % scale by alpha and 1/m.
        thetaChange = (alpha * gradients) .* (1 / m);
        theta = theta - thetaChange;

        J_history(iter) = computeCost(X, y, theta);
    end
end