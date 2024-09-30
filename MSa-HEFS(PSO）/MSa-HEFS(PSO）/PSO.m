function [finalAgents, scores] = PSO(mdl, initialAgents, maxIter, maxVelocity, U_n)
    agents = initialAgents;
    [nAgents, n_d] = size(initialAgents);
    fitnessValues = predict(mdl, agents);
    localBestPosition=[agents fitnessValues];
    for iter = 1:maxIter
   %% PSO produce new particles
     
        % Velocity update
        indices = randperm(nAgents,1); 
        pop = agents(indices(1), :);
        [~,ind]=max(fitnessValues);
        globalBestPosition=agents(ind);
        cognitiveComponent = rand * rand * (localBestPosition(indices,1:end-1) - pop);
        socialComponent = (1-rand) * rand * (globalBestPosition - pop);
        newVelocity =  cognitiveComponent + socialComponent;

        % Velocity Limit
        newVelocity(newVelocity > maxVelocity) = maxVelocity;
        newVelocity(newVelocity < -maxVelocity) = -maxVelocity;

        tempPosition = pop + newVelocity; 
        tempPosition = tempPosition ./ (2*maxVelocity) + 0.5; 
        newPop= tempPosition >= 0.5;
        newPop = newPop + 0; 
 
        newFitness = predict(mdl, newPop);
         if newFitness > fitnessValues(indices)
             localBestPosition(indices,:)=[newPop newFitness]; %Update local optimal solutions
             fitnessValues(indices)=newFitness;
         end
         if newFitness >  fitnessValues(ind)
             fitnessValues(ind)=newFitness;  %Update global optimal
             agents(ind,:)=newPop;
         end
        
        [~, worstIdx] = min(fitnessValues);
        if newFitness > fitnessValues(worstIdx)
            agents(worstIdx, :) = newPop;
            fitnessValues(worstIdx) = newFitness;
        end
    end
    
    finalAgents = agents;
    scores = fitnessValues;
end
