function scaledX = scaleData(X)
    minX = min(X);
    X = bsxfun(@minus, X, minX);

    rangeX = max(X) - minX;
    scaledX = bsxfun(@rdivide, X, rangeX);
end