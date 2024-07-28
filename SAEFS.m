function [Feature_y, DB] = SAEFS(X, y, Tmax)
    Feature_y = [];
    fisher_percent = 0.6;
    k_relief = 5; 
    m = 10;   %候选代理池规模
    N_latin = 50;    %拉丁超立方采样的个体数，即初始化种群规模
    X = scaleData(X);   
   
    rank1 = IG(X, y);
    rank2 = relieff(X, y, k_relief);
    rank3 = fscmrmr(X,y);
    
    ensemble_rank = rank1 + rank2 + rank3;
    [~, Indices] = sort(ensemble_rank, 'descend');
    n_feature_DB = floor(fisher_percent * size(X, 2));
    DB = X(:, Indices(1:n_feature_DB));

    selectedPoints = clusterSelection(DB,y, m);   
    
    Latin_sample_X = lhsdesign(N_latin,n_feature_DB);    

    [probabilities,~]  = get_acc(selectedPoints{1,m}(:,1:n_feature_DB), selectedPoints{1,m}(:,n_feature_DB+1),Latin_sample_X, 0.2);   %初始时使用规模最大的实例代理训练模型
    real_sample = selectedPoints{1,m};
    TD = [Latin_sample_X, probabilities];
   
    kriging_mdl = fitrgp(Latin_sample_X,probabilities);   
      

    T = 1;
    T_update = 30;       


    maxIter = 100; 
    alpha = 0.5; 
    beta = 0.5; 
       
    a = 0.1;    
    finalAgents = Latin_sample_X;

    percent_delete = 0.05;    
    currenct_best_score = 0;
    currenct_best_agent = TD(1,:);
    
    while T <= Tmax
        T;
        [finalAgents, scores] = BSO(kriging_mdl, finalAgents, maxIter, alpha, beta);    %利用克里金模型对所有个体acc进行预测
        [Scores, Indices] = sort(scores, 'descend');    
        sortedAgents = finalAgents(Indices, :);
        
        splitIndex = round(3 * a * size(sortedAgents, 1));    
        part1 = sortedAgents(1:splitIndex, :);   
        
        [part1_scores,~]  = get_acc(real_sample(:,1:n_feature_DB), real_sample(:,n_feature_DB+1), part1, 0.2);   %利用实例代理对前3a%的个体acc进行预测
        [Part1Scores, part1Indices] = sort(part1_scores, 'descend');
        sortedPart1 = part1(part1Indices, :);

        splitIndexPart1 = round(1/3 * size(sortedPart1, 1));   
        part1_1third = sortedPart1(1:splitIndexPart1, :);     
        
        [part1_1third_scores,F1]  = get_acc(DB, y, part1_1third, 0.2);  
        
        need_to_add = [part1_1third, part1_1third_scores];

        sortedTD = sortrows(TD, size(TD, 2));
        num_Remove = round(percent_delete * size(TD, 1));
        TD = sortedTD(num_Remove+1:end, :);      
   
        TD = [TD; need_to_add];      
        kriging_mdl = fitrgp(TD(:,1:n_feature_DB),TD(:,n_feature_DB+1));   


        if mod(T,T_update) == 0    

            kriging_scores = predict(kriging_mdl, finalAgents);
            
            [~, maxScoreIdx] = max(kriging_scores);
            bestAgentrg = finalAgents(maxScoreIdx, :);    

            [mdl_scores,~]  = get_acc(real_sample(:,1:n_feature_DB), real_sample(:,n_feature_DB+1), finalAgents, 0.2);
            
            [~, maxScoreIdx] = max(mdl_scores);
            bestAgentreal = finalAgents(maxScoreIdx, :);     
            
            [FS1,~]  = get_acc(DB, y, bestAgentrg, 0.2);
            [FS2,~]  =get_acc(DB, y, bestAgentreal, 0.2);    
            FS=FS1+FS2;
            FS_result = [];
            for i = 1:m    
                [new_part1,~]  = get_acc(selectedPoints{1,i}(:,1:n_feature_DB), selectedPoints{1,i}(:,n_feature_DB+1), bestAgentrg, 0.2);
                [new_part2,~]  = get_acc(selectedPoints{1,i}(:,1:n_feature_DB), selectedPoints{1,i}(:,n_feature_DB+1), bestAgentreal, 0.2);
                new_FS = new_part1 + new_part2;
                FS_result = [FS_result, new_FS];
            end
            FS_result_abs = abs(FS_result - FS);
            [~, mdl_idx] = min(FS_result_abs);
            real_sample = selectedPoints{1,mdl_idx};  
        end

        T = T + 1;

        [Scores, Indices] = sortrows([part1_1third_scores F1], 'descend');  
        sortedAgents = part1_1third(Indices, :);

        score = Scores(1,1);
        f1_score = Scores(1,2); 
        if score > currenct_best_score  
            
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

plot(1:1:Tmax,Feature_y(:,size(Feature_y,2)))

