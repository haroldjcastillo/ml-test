X = [1, 89, 7921;     1, 72, 5184;     1, 94, 8836;     1, 69, 4761]y = [96; 74; 87; 78]% max(X, [], 1) the max value of each column% max(X, [], 2) the max value of each rowmx = max(X(:, 3))mn = min(X(:, 3))avg = sum(X(:, 3))/length(X(:, 3))theta = X(4, 3) - avg/ mx - mn