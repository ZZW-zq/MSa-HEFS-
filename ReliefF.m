function rank = ReliefF(X, y, k)
    
    [num, numFeatures] = size(X);
    weights = zeros(1, numFeatures);

    for i = 1:num
        instance = X(i, :);
        distances = sum((X - instance).^2, 2);  

        hitIndices = find(y == y(i) & distances > 0);
        [~, sortedHitIndices] = sort(distances(hitIndices));
        hits = hitIndices(sortedHitIndices(1:min(k, length(sortedHitIndices))));

        missIndices = find(y ~= y(i));
        [~, sortedMissIndices] = sort(distances(missIndices));
        misses = missIndices(sortedMissIndices(1:min(k, length(sortedMissIndices))));

        for j = 1:numFeatures
            weights(j) = weights(j) - sum((instance(j) - X(hits, j)).^2) + sum((instance(j) - X(misses, j)).^2);
        end
    end

    [~, rank] = sort(weights, 'descend');
end
