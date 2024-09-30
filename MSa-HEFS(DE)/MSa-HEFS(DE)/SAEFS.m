function [Feature_y, DB] = SAEFS(X, y, Tmax)
    Feature_y = [];
    fisher_percent = 0.6;
    k_relief = 5;  %k-nearest neighbor parameters for reliefF
    m = 10;   %the size of candidate surrogate pool size
    N_latin = 50;    %The number of individuals in the Latin hypercube sample
    X = scaleData(X);    %data normalization
   
    rank1 = IG(X, y);
    rank2 = relieff(X, y, k_relief);
    rank3 = fscmrmr(X,y);
    
    ensemble_rank = rank1 + rank2 + rank3;
    [~, Indices] = sort(ensemble_rank, 'descend');
    n_feature_DB = floor(fisher_percent * size(X, 2));
    DB = X(:, Indices(1:n_feature_DB));

    selectedPoints = clusterSelection(DB,y, m);   %Build an instance surrogate pool with the size of m=10
    
    Latin_sample_X = lhsdesign(N_latin,n_feature_DB);     %Latin hypercube sampling

    [probabilities,~]  = get_acc(selectedPoints{1,m}(:,1:n_feature_DB), selectedPoints{1,m}(:,n_feature_DB+1),Latin_sample_X, 0.2);   %Start with the largest instance surrogate to train the model
    real_sample = selectedPoints{1,m};
    TD = [Latin_sample_X, probabilities];
   
    kriging_mdl = fitrgp(Latin_sample_X,probabilities);    %train the kriging model
      
%Start iteration
    T = 1;
    T_update = 30;       %The instance surrogate is updated every T_update generation

%parameters for DE algorithm
    maxIter = 100; 
       
    a = 0.1;     % Individual hierarchical evaluation mechanism parameters, that is, 3a%
    finalAgents = Latin_sample_X;

    percent_delete = 0.05;    %Remove the proportion of poor individuals in TD
    currenct_best_score = 0;
    currenct_best_agent = TD(1,:);

    while T <= Tmax
        T;
        [finalAgents, scores] = SaWDE(kriging_mdl, finalAgents, maxIter);
[Scores, Indices] = sort(scores, 'descend');     %Sort acc from largest to smallest
        sortedAgents = finalAgents(Indices, :);
        
        splitIndex = round(3 * a * size(sortedAgents, 1));    
        part1 = sortedAgents(1:splitIndex, :);   %Select the top 3a% of individuals
        
        [part1_scores,~]  = get_acc(real_sample(:,1:n_feature_DB), real_sample(:,n_feature_DB+1), part1, 0.2);   %The instance surrogate is used to predict the acc of the first 3a% of individual
        [Part1Scores, part1Indices] = sort(part1_scores, 'descend');
        sortedPart1 = part1(part1Indices, :);

        splitIndexPart1 = round(1/3 * size(sortedPart1, 1));   
        part1_1third = sortedPart1(1:splitIndexPart1, :);     %Select the first 1/3 individuals
        
        [part1_1third_scores,F1]  = get_acc(DB, y, part1_1third, 0.2);    %evaluate the acc of the top 1/3 individuals with all samples
        
        need_to_add = [part1_1third, part1_1third_scores];

        sortedTD = sortrows(TD, size(TD, 2));
        num_Remove = round(percent_delete * size(TD, 1));
        TD = sortedTD(num_Remove+1:end, :);      %remove the poorer individuals in TD Proportionally
   
        TD = [TD; need_to_add];       %Update the sample set of kriging model
        kriging_mdl = fitrgp(TD(:,1:n_feature_DB),TD(:,n_feature_DB+1));     %Retrain the kriging model

%%%%%%%%%%%%%%%%%Update the instance surrogate model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if mod(T,T_update) == 0    %Select a new instance surrogate from the candidate sample surrogate pool every T_update generation

            kriging_scores = predict(kriging_mdl, finalAgents);
            
            [~, maxScoreIdx] = max(kriging_scores);
            bestAgentrg = finalAgents(maxScoreIdx, :);    %Record the best individual predicted by the kriging model

            [mdl_scores,~]  = get_acc(real_sample(:,1:n_feature_DB), real_sample(:,n_feature_DB+1), finalAgents, 0.2);
            
            [~, maxScoreIdx] = max(mdl_scores);
            bestAgentreal = finalAgents(maxScoreIdx, :);     %Record the best individual by the instances surrogate 
            
            [FS1,~]  = get_acc(DB, y, bestAgentrg, 0.2);
            [FS2,~]  =get_acc(DB, y, bestAgentreal, 0.2);    %evaluate the above two individuals with all instances
            FS=FS1+FS2;
            FS_result = [];
            for i = 1:m    %evaluate the above two individuals separately with all instance surrogates in the surrogate pool
                [new_part1,~]  = get_acc(selectedPoints{1,i}(:,1:n_feature_DB), selectedPoints{1,i}(:,n_feature_DB+1), bestAgentrg, 0.2);
                [new_part2,~]  = get_acc(selectedPoints{1,i}(:,1:n_feature_DB), selectedPoints{1,i}(:,n_feature_DB+1), bestAgentreal, 0.2);
                new_FS = new_part1 + new_part2;
                FS_result = [FS_result, new_FS];
            end
            FS_result_abs = abs(FS_result - FS);
            [~, mdl_idx] = min(FS_result_abs);
            real_sample = selectedPoints{1,mdl_idx};   %select the instance surrogate with the smallest difference as the instance surrogate of the subsequent evaluation individual
        end

        T = T + 1;

        [Scores, Indices] = sortrows([part1_1third_scores F1], 'descend');   %rank the acc of the truly evaluated individuals in descending order
        sortedAgents = part1_1third(Indices, :);

        score = Scores(1,1);
        f1_score = Scores(1,2); 
        if score > currenct_best_score   %Output the result of each generation of feature selection and save it in Feature_y
            
            currenct_best_score = score;
            bestData = sortedAgents(1, :);
            
            currenct_best_agent = double(bestData>0.5);
            need_to_add_1 = [currenct_best_agent, score,f1_score];
            Feature_y = [Feature_y; need_to_add_1];
        else
            need_to_add_1 = [currenct_best_agent, currenct_best_score,f1_score];
            Feature_y = [Feature_y; need_to_add_1];
        
        end 
    end


