function [accuracy,F1] = get_acc(X, y, feature, ratio)
    accuracy = [];
    F1=[]; 
    [m,~] = size(feature);
    for j = 1:m
        new_X = X(:,(feature(j,:)>0.5));
        cv = cvpartition(size(X, 1), 'HoldOut', ratio);
    
        X_train = new_X(training(cv), :);
        y_train = y(training(cv), :);

        X_test = new_X(test(cv), :);
        y_test = y(test(cv), :);
        mdl = fitcknn(X_train, y_train,'NumNeighbors',3);
        predict_y = predict(mdl, X_test);
        correct = sum(y_test == predict_y);
        
        confMat = confusionmat(y_test', predict_y');

        numClasses = size(confMat, 1);

        TP = zeros(numClasses, 1);
        FP = zeros(numClasses, 1);
        FN = zeros(numClasses, 1);

        for i = 1:numClasses
            TP(i) = confMat(i,i); 
            FP(i) = sum(confMat(:,i)) - TP(i); 
            FN(i) = sum(confMat(i,:)) - TP(i); 
        end
  

        Precision = zeros(numClasses, 1);
        Recall = zeros(numClasses, 1);
        F1Score = zeros(numClasses, 1);


        for i = 1:numClasses
            Precision(i) = TP(i) / (TP(i) + FP(i)); 
            Recall(i) = TP(i) / (TP(i) + FN(i));   
        end

        for i = 1:numClasses
            F1Score(i) = 2 * (Precision(i) * Recall(i)) / (Precision(i) + Recall(i)); 
        end


        sum_nor=size(F1Score,1)-sum(isnan(F1Score));
        F1Score(isnan(F1Score))=0;
        macroF1 = sum(F1Score)/sum_nor;
        F1 =[F1;  macroF1];
        accuracy =[accuracy; correct / length(y_test)];
    end


end