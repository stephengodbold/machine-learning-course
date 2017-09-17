function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


%remove the theta zero case for theta and X
thetaOther = theta(2:end, :);
XOther =  X(:,2:end);

%calculate the cost
sig = sigmoid(X*theta);
oneCase = -y .* log(sig);
zeroCase = (1-y) .* log(1-sig);
costReg = (lambda/(2*m)) * (sum(thetaOther .^ 2));

J =  ((1/m) * sum(oneCase - zeroCase)) + costReg;

%calculate for theta zero
XZero = X(end,1);
thetaZero = theta(end,1);
sigZero = sigmoid(XZero * thetaZero);
gradZero = (1/m) .* sum((sig - y) .* XZero);

%calculate for theta one through n
gradAll = (1/m) .* sum((sig - y) .* XOther) + ((lambda/m) .* thetaOther');

%recombine the gradient matrix
grad = [gradZero, gradAll];
% =============================================================

grad = grad(:);

end
