function [finalAgents, scores] = ImprovedGA(mdl, initialAgents, maxIter)
    % IGA Parameters
    popSize = size(initialAgents, 1);
    nFeatures = size(initialAgents, 2);
    q = 20; % number of controlled individuals
    f =10;  %number of controlled individuals

    agents = initialAgents;
    scores = predict(mdl, agents);

    for iter = 1:maxIter
        % Sort agents by their fitness (accuracy)
        [~, sortedIdx] = sort(scores, 'descend');
        agents = agents(sortedIdx, :);
        scores = scores(sortedIdx);

        % Apply Elitism - Retain the best solutions
        newAgents = agents(1:q, :);
        
        % Adaptive adjustment of crossover and mutation rates
        avgFitness = mean(scores);
        bestFitness = max(scores);
        crossoverRate = 0.8 * (1 - (bestFitness - avgFitness) / bestFitness);
        mutationRate = 0.1 * (1 - (bestFitness - avgFitness) / bestFitness);

        % Generate new population
        while size(newAgents, 1) < popSize
            % Selection - Roulette Wheel
            parent1 = agents(RouletteWheelSelection(scores), :);
            parent2 = agents(RouletteWheelSelection(scores), :);

            % Crossover
            if rand < crossoverRate
                crossPoint = randi([1, nFeatures - 1]);
                child1 = [parent1(1:crossPoint), parent2(crossPoint + 1:end)];
                child2 = [parent2(1:crossPoint), parent1(crossPoint + 1:end)];
            else
                child1 = parent1;
                child2 = parent2;
            end

            % Mutation
            for j = 1:nFeatures
                if rand < mutationRate
                    child1(j) = 1 - child1(j); % Flip bit
                end
                if rand < mutationRate
                    child2(j) = 1 - child2(j); % Flip bit
                end
            end

            % Add children to the new population
            newAgents = [newAgents; child1; child2];
        end

        % Evaluate new population
        newAgents = newAgents(1:popSize, :); % Ensure population size
        scores = predict(mdl, newAgents);
    end

    finalAgents = agents;
end

function selectedIndex = RouletteWheelSelection(scores)
    % Roulette Wheel Selection based on fitness
    cumulativeSum = cumsum(scores);
    totalSum = sum(scores);
    randPoint = rand * totalSum;
    selectedIndex = find(cumulativeSum >= randPoint, 1, 'first');
end
