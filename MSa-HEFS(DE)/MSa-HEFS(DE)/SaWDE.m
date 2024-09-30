function [finalAgents, scores] = SaWDE(mdl, initialAgents, maxIter)
    % SaWDE Parameters
    numSubPop = 5; % Number of sub-populations
    F_pool = [0.5, 1, 0.6, 0.9, 0.5, 0.9, 0.6, 1]; % Scaling factors
    CR_pool = [0.1, 0.2, 0.9, 0.8, 0.9, 0.1, 0.8, 0.2]; % Crossover probabilities
    maxGenerations = maxIter;
    popSize = size(initialAgents, 1);
    
    % Divide initial population into sub-populations
    subPopSize = ceil(popSize / numSubPop);
    subPops = mat2cell(initialAgents, repmat(subPopSize, 1, numSubPop), size(initialAgents, 2));

    % Initialize fitness scores
    scores = predict(mdl, initialAgents);
    fitnessHistory = scores;
    
    % Initialize strategy pool
    strategyPool = {'DE/current to best/1', 'DE/current to rand/1', 'DE/rand/3', 'DE/best/1', ...
                    'DE/rand to best/1', 'DE/rand/2', 'DE/best/2', 'DE/best/3'};
    strategyCount = zeros(1, length(strategyPool));
    strategySuccess = zeros(1, length(strategyPool));
    
    for generation = 1:maxGenerations
        % For each sub-population
        for i = 1:numSubPop
            subPop = subPops{i};
            % Select a strategy from the pool
            strategyIdx = randi(length(strategyPool));
            strategy = strategyPool{strategyIdx};
            F = F_pool(strategyIdx);
            CR = CR_pool(strategyIdx);
            
            % Select the best agent based on predicted accuracy
            [~, bestIdx] = max(predict(mdl, subPop)); % Find the best individual based on the prediction
            x_best = subPop(bestIdx, :);

            for j = 1:subPopSize
                % Select agents for mutation
                if contains(strategy, '3')  % Strategies requiring 7 random indices
                    r = randperm(subPopSize, 7);
                elseif contains(strategy, '2')  % Strategies requiring 5 random indices
                    r = randperm(subPopSize, 5);
                else  % Strategies requiring 3 random indices
                    r = randperm(subPopSize, 3);
                end

                x1 = subPop(r(1), :);
                x2 = subPop(r(2), :);
                x3 = subPop(r(3), :);

                % Generate mutant vector based on selected strategy
                switch strategy
                    case 'DE/current to best/1'
                        mutant = subPop(j, :) + F * (x_best - subPop(j, :)) + F * (x2 - x3);
                    case 'DE/current to rand/1'
                        mutant = subPop(j, :) + rand * (x1 - subPop(j, :)) + F * (x2 - x3);
                    case 'DE/rand/3'
                        mutant = x1 + F * (x2 - x3 + subPop(r(4), :) - subPop(r(5), :) + subPop(r(6), :) - subPop(r(7), :));
                    case 'DE/best/1'
                        mutant = x_best + F * (x2 - x3);
                    case 'DE/rand to best/1'
                        mutant = x1 + F * (x_best - subPop(j, :)) + F * (x2 - x3);
                    case 'DE/rand/2'
                        mutant = x1 + F * (x2 - x3) + F * (subPop(r(4), :) - subPop(r(5), :));
                    case 'DE/best/2'
                        mutant = x_best + F * (x2 - x3) + F * (subPop(r(4), :) - subPop(r(5), :));
                    case 'DE/best/3'
                        mutant = x_best + F * (x2 - x3 + subPop(r(4), :) - subPop(r(5), :) + subPop(r(6), :) - subPop(r(7), :));
                end
                
                % Ensure mutant is within bounds
                mutant = max(0, min(1, mutant));
                
                % Crossover
                crossPoints = rand(size(mutant)) < CR;
                trial = subPop(j, :);
                trial(crossPoints) = mutant(crossPoints);
                
                % Selection
                trialFitness = predict(mdl, trial);
                if trialFitness > scores(j)
                    subPop(j, :) = trial;
                    scores(j) = trialFitness;
                    strategySuccess(strategyIdx) = strategySuccess(strategyIdx) + 1; % Success counter
                end
                strategyCount(strategyIdx) = strategyCount(strategyIdx) + 1; % Total count
            end
            
            % Update sub-population
            subPops{i} = subPop;
        end
        
        % Update strategy pool by success rate
        if mod(generation, 20) == 0
            successRate = strategySuccess ./ strategyCount;
            [~, topStrategies] = sort(successRate, 'descend');
            strategyPool = strategyPool(topStrategies(1:5)); % Keep top 5 strategies
            strategyCount = zeros(1, length(strategyPool)); % Reset counts
            strategySuccess = zeros(1, length(strategyPool));
        end
    end
    
    % Combine sub-populations back into the main population
    finalAgents = cell2mat(subPops);
    scores = predict(mdl, finalAgents); % Recalculate scores for the final agents
end