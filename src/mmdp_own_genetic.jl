begin
  using Graphs
  using SimpleWeightedGraphs
  using Folds
  using Base.Iterators
  using Random
end

include("graph_utils.jl")

function maxmindp_random(n, k)
  randperm(n)[1:k]
end

struct GeneticSettings
  populationSize
  mutationRate
  crossoverRate
  elitRate
  crossoverAlg
  mutationAlg
end

struct RunSettings
  minDists
  k
  numberOfIterations
  logging
  RunSettings(minDists, k, numberOfIterations) = new(minDists, k, numberOfIterations, "")
  RunSettings(minDists, k, numberOfIterations, logging) = new(minDists, k, numberOfIterations, logging)
end


function maxmindp_genetic(runS::RunSettings, gaS::GeneticSettings, chromosomes)
  # Initializing values and functions for later use
  n = length(chromosomes)
  calcFitness(x) = calculate_mindist(x, runS.minDists)
  runMutation(x) = rand() < gaS.mutationRate ? gaS.mutationAlg(n, x, runS.minDists) : x
  chromosomes = deepcopy(chromosomes)

  # Initializing global maximum as one of the given chromosome
  maxVal = calculate_mindist(chromosomes[1], runS.minDists)
  maxVec = copy(chromosomes[1])

  # Initializing logging
  logs = []

  fitness = collect(
    map(calcFitness, chromosomes)
  )
  for i in 1:runS.numberOfIterations
    # Creating p_c% new individuals with the crossover
    # operator, choosing parents based on fitness.
    newChromosomes = [
      gaS.crossoverAlg(chromosomes, fitness, runS.minDists)
      for _ in 1:Int(ceil(n * gaS.crossoverRate))
    ]
    newFitness = collect(
      map(calcFitness, chromosomes)
    )

    # Add them to the chromosome pool
    append!(chromosomes, newChromosomes)
    append!(fitness, newFitness)

    # Mutating individuals
    chromosomes = collect(
      map(runMutation, chromosomes)
    )

    # Recalculating fitness for new individuals
    fitness = collect(
      map(calcFitness, chromosomes)
    )

    # Sorting fitness scores
    fitnessSorted = sortperm(fitness, rev=true)

    fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
    fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


    # Choosing the elit
    elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
    elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
    elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

    # Choosing the rest randomly from the others
    restNumber = gaS.populationSize - elitNumber
    restIds = [
      rand(fitnessSorted[elitNumber+1:end])
      for _ in 1:restNumber
    ]
    restChromosomes = map(x -> copy(chromosomes[x]), restIds)
    restFitness = map(x -> fitness[x], restIds)

    chromosomes = vcat(elitChromosomes, restChromosomes)
    fitness = vcat(elitFitness, restFitness)

    if fitnessMaxVal > maxVal
      maxVec = deepcopy(fitnessMaxVec)
      maxVal = deepcopy(fitnessMaxVal)
    end

    if runS.logging != ""
      logRow = [i, maxVal]
      append!(logRow, sort(maxVec))
      push!(logs, logRow)
    end
  end

  maxVec, logs
end
