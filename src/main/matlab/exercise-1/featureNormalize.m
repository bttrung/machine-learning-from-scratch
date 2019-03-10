function [X_norm, mu, sigma] = featureNormalize(X)

X_norm = X;

mu = mean(X);
sigma = std(X);

% for each X column (features)
for col = 1:length(X(1,:))
    for row = 1:length(X(:,1))
        X_norm(row,col) = (X (row,col) - mu(1,col))/sigma(1,col)
    end
end

end
