module RandomBest

using Random

function randomBest(minDists, k, randVals, fitF=DispersionProblems.calculateMinSumdp)
    n, _ = size(minDists)
    options = [randperm(n)[1:k] for _ = 1:randVals]

    fitness = [fitF(x, minDists) for x in options]

    best = argmax(fitness)

    options[best], fitness[best]
end

end