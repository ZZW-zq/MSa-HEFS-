function [finalAgents, scores] = BSO(mdl, initialAgents, maxIter, alpha, beta)
    
    agents = initialAgents;
    [nAgents, n_d] = size(initialAgents);
    

    fitnessValues = predict(mdl, agents);
    
    for iter = 1:maxIter
        indices = randperm(nAgents, 2); 
        agent1 = agents(indices(1), :);
        agent2 = agents(indices(2), :);
        
        newIdea = (1 - alpha) * agent1 + alpha * agent2 + beta * (rand(1, size(agents, 2)) - 0.5);
        
        newIdea = max(0, min(1, newIdea));  

        for j=1:n_d
            ax = 1+exp(5-10*newIdea(1,j));
            if 1/ax>rand
                newIdea(1,j) = 1;
            else
                newIdea(1,j) = 0;
            end
        end
        
        newFitness = predict(mdl, newIdea);
        
        [~, worstIdx] = min(fitnessValues);
        if newFitness > fitnessValues(worstIdx)
            agents(worstIdx, :) = newIdea;
            fitnessValues(worstIdx) = newFitness;
        end
    end
    
    finalAgents = agents;
    scores = fitnessValues;
end
