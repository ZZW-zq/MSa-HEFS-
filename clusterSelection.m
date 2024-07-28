function selectedPoints = clusterSelection(DB, y, m)

    [num, n] = size(DB); 
    km = num / (2 * n);
    
    selectedPoints = cell(1, m);   

    for i = 1:m
        k = ceil(i * n * km / m + 1);
        [idx, C] = kmeans(DB, k); 

        Points = zeros(k, n); 
        Labels = zeros(k, 1); 

        for j = 1:k   
            clusterData = DB(idx == j, :); 
            clusterLabels = y(idx == j); 
            distances = sum((clusterData - C(j, :)).^2, 2);
            [~, minIndex] = min(distances); 
            Points(j, :) = clusterData(minIndex, :); 
            Labels(j) = clusterLabels(minIndex);   
        end

        selectedPoints{i} = [Points, Labels]; 
    end
end
