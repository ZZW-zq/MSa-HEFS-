function selectedPoints = clusterSelection(DB, y, m)

    [num, n] = size(DB); 
    km = num / (2 * n);
    
    selectedPoints = cell(1, m);   %Initializes an array of cells, storing selected data points for each k-means iteration

    for i = 1:m
        k = ceil(i * n * km / m + 1);
        [idx, C] = kmeans(DB, k); 

        Points = zeros(k, n); 
        Labels = zeros(k, 1); 

        for j = 1:k   %For each candidate surrogate pool, select k clusters (instance surrogates)
            clusterData = DB(idx == j, :); 
            clusterLabels = y(idx == j); 
            distances = sum((clusterData - C(j, :)).^2, 2); % Square of the Euclidean distance
            [~, minIndex] = min(distances); 
            Points(j, :) = clusterData(minIndex, :); % Selects the nearest point of the current cluster
            Labels(j) = clusterLabels(minIndex);    % Select the label corresponding to the nearest point
        end

        selectedPoints{i} = [Points, Labels]; 
    end
end
