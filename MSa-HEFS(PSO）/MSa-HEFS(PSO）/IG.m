function rank = IG(X, y)
    H_y = entropy(y);
    numFeatures = size(X, 2);
    ig = zeros(1, numFeatures);

    for i = 1:numFeatures
        H_y_given_x = conditional_entropy(X(:, i), y);
        ig(i) = H_y - H_y_given_x;
    end
   
    [~, rank] = sort(ig, 'descend');
end

function H = entropy(y)
    labels = unique(y);
    H = 0;
    for i = 1:length(labels)
        p = sum(y == labels(i)) / length(y);
        H = H - p * log2(p);
    end
end

function H = conditional_entropy(x, y)
    uniqueX = unique(x);
    H = 0;
    for i = 1:length(uniqueX)
        idx = x == uniqueX(i);
        H = H + (sum(idx) / length(x)) * entropy(y(idx));
    end
end
