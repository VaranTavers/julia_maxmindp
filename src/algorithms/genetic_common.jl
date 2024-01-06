struct GeneticSettings
    populationSize::Integer
    mutationRate::Float64
    crossoverRate::Float64
    elitRate::Float64
    crossoverAlg::Any
    mutationAlg::Any
end

struct RunSettings
    minDists::Any
    k::Integer
    numberOfIterations::Integer
    logging::Bool
    RunSettings(minDists, k, numberOfIterations) = new(minDists, k, numberOfIterations, "")
    RunSettings(minDists, k, numberOfIterations, logging) =
        new(minDists, k, numberOfIterations, logging)
end
