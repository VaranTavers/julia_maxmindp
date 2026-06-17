module Genetic

using ..DispersionProblems
using Graphs
using SimpleWeightedGraphs
using Folds
using Base.Iterators
using Random
using Statistics

include("../utils/graph_utils.jl")

function get_candidates(n, v)
    a = collect(1:n)

    collect(setdiff(Set(a), Set(v)))
end


include("../crossovers/naive.jl")
include("../crossovers/sbts_like.jl")
include("../crossovers/sbts_sane.jl")
include("../crossovers/greedy_like.jl")
include("../crossovers/inception.jl")

include("../crossover_selection/random.jl")
include("../crossover_selection/fitness_roulette.jl")

include("../mutations/random.jl")
include("../mutations/sbts_sane.jl")
include("../mutations/sbts.jl")
include("../mutations/change_to_close.jl")
include("../mutations/combiner.jl")

include("../inits/random_init.jl")
include("../inits/high_sum.jl")


struct GeneticSettings
    populationSize::Integer
    mutationRate::Float64
    crossoverRate::Float64
    elitRate::Float64
    crossoverAlg::Function
    crossoverSelAlg::Function
    mutationAlg::Function
    initAlg::Function
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

function maxmindp_genetic(
    runS::RunSettings,
    gaS::GeneticSettings;
    fitF=DispersionProblems.calculateMinSumdp,
    partFitF=DispersionProblems.calculateSumdp
)
    # Initializing values and functions for later use
    n, _ = size(runS.minDists)
    stats = zeros((n, runS.numberOfIterations))
    avgFit = zeros((n, runS.numberOfIterations))
    numberOfPoints, _ = size(runS.minDists)
    calcFitness(x) = maximum(fitF(x, runS.minDists))
    runMutation(x) =
        rand() < gaS.mutationRate ? gaS.mutationAlg(numberOfPoints, x, runS.minDists, partFitF) : x
    runCrossover(left, right) = gaS.crossoverAlg(left, right, runS.minDists, partFitF)

    chromosomes = gaS.initAlg(gaS.populationSize, runS.k, runS.minDists)



    # Initializing global maximum as one of the given chromosome
    maxVal = calcFitness(chromosomes[1])
    maxVec = copy(chromosomes[1])

    # Initializing logging
    logs = []

    fitness = collect(map(calcFitness, chromosomes))
    for i = 1:runS.numberOfIterations
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        newChromosomes = [
            gaS.crossoverSelAlg(chromosomes, fitness, runCrossover) for
            _ = 1:Int(ceil(gaS.populationSize * gaS.crossoverRate))
        ]
        newFitness = collect(map(calcFitness, newChromosomes)) #TODO: FIX this and test!!!!

        # Add them to the chromosome pool
        append!(chromosomes, newChromosomes)
        append!(fitness, newFitness)

        # Mutating individuals
        chromosomes = collect(map(runMutation, chromosomes))

        # Recalculating fitness for new individuals
        fitness = collect(map(calcFitness, chromosomes))

        if runS.logging != ""
            for (chr, fit) in zip(chromosomes, fitness)
                for node in chr
                    stats[node, i] += 1
                    avgFit[node, i] += fit
                end
            end
        end

        # Sorting fitness scores
        fitnessSorted = sortperm(fitness, rev=true)

        fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
        fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


        # Choosing the elit
        elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
        elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
        elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

        #=
        # Choosing the rest randomly from the others
        restNumber = gaS.populationSize - elitNumber
        restIds = [rand(fitnessSorted[elitNumber+1:end]) for _ = 1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)

        =#

        # Choosing the rest with diversity in mind 
        restNumber = gaS.populationSize - elitNumber
        avgDifferentNodes = [
            mean(
                [length(chromosomes[i]) - count(chromosomes[i] .∈ j) for j in elitChromosomes]
            )
            for i in fitnessSorted[elitNumber+1:end]
        ]
        sortedAvgDiff = sortperm(avgDifferentNodes, rev=true)
        restIds = fitnessSorted[elitNumber.+sortedAvgDiff][1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)


        chromosomes = vcat(elitChromosomes, restChromosomes)
        fitness = vcat(elitFitness, restFitness)



        if fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
        end

        if runS.logging != ""
            logRow = stats[:, i][:]
            append!(logRow, avgFit[:, i][:] ./ stats[:, i][:])
            append!(logRow, [i, maxVal])
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
    end


    maxVec, maxVal, logs
end

function customMutationAlg2(n, v, minDists, numOfMoves)
    @assert numOfMoves <= length(v) && numOfMoves > 0

    toChange = collect(1:length(v))
    shuffle!(toChange)

    for i in toChange[1:numOfMoves]
        good = false
        j = 0
        while !good
            j = rand(1:n)
            good = minDists[v[i], j] > 0 && findfirst(x -> x == j, v) === nothing
        end
        v[i] = j
    end

    v
end

function maxmindp_genetic_tree(runS, gaS, chromosomes;
    fitF=DispersionProblems.calculateMinSumdp, partFitF=DispersionProblems.calculateSumdp)
    # Initializing values and functions for later use
    n = length(chromosomes)
    numberOfPoints, _ = size(runS.minDists)
    calcFitness(x) = maximum(fitF(x, runS.minDists))
    runMutation(x) =
        rand() < gaS.mutationRate ? gaS.mutationAlg(numberOfPoints, x, runS.minDists, partFitF) : x
    chromosomes = deepcopy(chromosomes)

    # Initializing global maximum as one of the given chromosome
    maxVal = calculate_mindist(chromosomes[1], runS.minDists)
    maxVec = copy(chromosomes[1])
    maxNum = 0

    # Initializing logging
    logs = []

    fitness = collect(map(calcFitness, chromosomes))
    for i = 1:runS.numberOfIterations
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        newChromosomes = [
            gaS.crossoverAlg(chromosomes, fitness, runS.minDists, partFitF) for
            _ = 1:Int(ceil(n * gaS.crossoverRate))
        ]
        newFitness = zeros(length(newChromosomes))


        # Add them to the chromosome pool
        append!(chromosomes, newChromosomes)
        append!(fitness, newFitness)

        # Mutating individuals
        chromosomes = collect(map(runMutation, chromosomes))

        # If the tree grew big enough it begins producing its offspring
        # that is in its neighborhood (the possible neighborhood grows as
        # the tree grows)
        if maxNum > 2
            # @show "Tree :)"
            treeChromosomes = [
                customMutationAlg2(
                    numberOfPoints,
                    deepcopy(maxVec),
                    runS.minDists,
                    min(length(maxVec) - 2, maxNum - 2),
                ) for _ = 1:Int(ceil(n * gaS.crossoverRate))
            ]

            treeFitness = zeros(length(treeChromosomes))

            # Add them to the chromosome pool
            append!(chromosomes, treeChromosomes)
            append!(fitness, treeFitness)

        end

        # Recalculating fitness for new individuals
        fitness = collect(map(calcFitness, chromosomes))

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
        restIds = [rand(fitnessSorted[elitNumber+1:end]) for _ = 1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)

        chromosomes = vcat(elitChromosomes, restChromosomes)
        fitness = vcat(elitFitness, restFitness)

        if fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
            maxNum = 0
        else
            maxNum += 1
        end

        if runS.logging != ""
            logRow = [i, maxVal]
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
    end

    maxVec, maxVal
end




function dual_genetic(
    runS::RunSettings,
    gaS::GeneticSettings;
    fitF=DispersionProblems.calculateMinSumdp,
    partFitF=DispersionProblems.calculateSumdp
)
    # Initializing values and functions for later use
    n, _ = size(runS.minDists)
    stats = zeros((n, runS.numberOfIterations))
    avgFit = zeros((n, runS.numberOfIterations))
    numberOfPoints, _ = size(runS.minDists)
    calcFitness(x) = maximum(fitF(x, runS.minDists))
    runMutation(x) =
        rand() < gaS.mutationRate ? gaS.mutationAlg(numberOfPoints, x, runS.minDists, partFitF) : x
    runCrossover1(left, right) = gaS.crossoverAlg(left, right, runS.minDists, partFitF)
    runCrossover2(left, right) = crossoverNaive(left, right, runS.minDists, partFitF)

    chromosomes = gaS.initAlg(gaS.populationSize, runS.k, runS.minDists)



    # Initializing global maximum as one of the given chromosome
    maxVal = calcFitness(chromosomes[1])
    maxVec = copy(chromosomes[1])

    # Initializing logging
    logs = []

    fitness = collect(map(calcFitness, chromosomes))
    for i = 1:runS.numberOfIterations
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        newChromosomes = [
            gaS.crossoverSelAlg(chromosomes, fitness, i < runS.numberOfIterations / 3 ? runCrossover1 : runCrossover2) for
            _ = 1:Int(ceil(gaS.populationSize * gaS.crossoverRate))
        ]
        newFitness = collect(map(calcFitness, newChromosomes)) #TODO: FIX this and test!!!!

        # Add them to the chromosome pool
        append!(chromosomes, newChromosomes)
        append!(fitness, newFitness)

        # Mutating individuals
        chromosomes = collect(map(runMutation, chromosomes))

        # Recalculating fitness for new individuals
        fitness = collect(map(calcFitness, chromosomes))

        if runS.logging != ""
            for (chr, fit) in zip(chromosomes, fitness)
                for node in chr
                    stats[node, i] += 1
                    avgFit[node, i] += fit
                end
            end
        end

        # Sorting fitness scores
        fitnessSorted = sortperm(fitness, rev=true)

        fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
        fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


        # Choosing the elit
        elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
        elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
        elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

        #=
        # Choosing the rest randomly from the others
        restNumber = gaS.populationSize - elitNumber
        restIds = [rand(fitnessSorted[elitNumber+1:end]) for _ = 1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)

        =#

        # Choosing the rest with diversity in mind 
        restNumber = gaS.populationSize - elitNumber
        avgDifferentNodes = [
            mean(
                [length(chromosomes[i]) - count(chromosomes[i] .∈ j) for j in elitChromosomes]
            )
            for i in fitnessSorted[elitNumber+1:end]
        ]
        sortedAvgDiff = sortperm(avgDifferentNodes, rev=true)
        restIds = fitnessSorted[elitNumber.+sortedAvgDiff][1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)


        chromosomes = vcat(elitChromosomes, restChromosomes)
        fitness = vcat(elitFitness, restFitness)



        if fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
        end

        if runS.logging != ""
            logRow = stats[:, i][:]
            append!(logRow, avgFit[:, i][:] ./ stats[:, i][:])
            append!(logRow, [i, maxVal])
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
    end


    maxVec, maxVal, logs
end

function perturb(mat)
    maxMat = maximum(mat)

    value = rand(1:20) / 100

    rands = rand(Float64, size(mat))

    newVals = (rands .< value) .* maxMat

    #@show newVals
    #@show mat

    @show count(newVals .> 0)

    mat + newVals
end

function maxmindp_genetic_perturbations(
    runS::RunSettings,
    gaS::GeneticSettings;
    fitF=DispersionProblems.calculateMinSumdp,
    partFitF=DispersionProblems.calculateSumdp,
    turnLength=200
)
    # Initializing values and functions for later use
    n, _ = size(runS.minDists)
    stats = zeros((n, runS.numberOfIterations))
    avgFit = zeros((n, runS.numberOfIterations))
    numberOfPoints, _ = size(runS.minDists)

    chromosomes = gaS.initAlg(gaS.populationSize, runS.k, runS.minDists)

    actualMinDists = runS.minDists

    # Initializing global maximum as one of the given chromosome
    maxVal = maximum(fitF(chromosomes[1], actualMinDists))
    maxVec = copy(chromosomes[1])

    # Initializing logging
    logs = []

    fitness = collect(map(x -> maximum(fitF(x, actualMinDists)), chromosomes))
    isPerturbed = true
    for i = 1:runS.numberOfIterations
        if i % turnLength == 1
            isPerturbed = !isPerturbed
            if isPerturbed
                actualMinDists = perturb(runS.minDists)
            else
                actualMinDists = copy(runS.minDists)
            end
        end
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        newChromosomes = [
            gaS.crossoverSelAlg(chromosomes, fitness, (left, right) -> gaS.crossoverAlg(left, right, actualMinDists, partFitF)) for
            _ = 1:Int(ceil(gaS.populationSize * gaS.crossoverRate))
        ]
        newFitness = collect(map(x -> maximum(fitF(x, actualMinDists)), newChromosomes)) #TODO: FIX this and test!!!!

        # Add them to the chromosome pool
        append!(chromosomes, newChromosomes)
        append!(fitness, newFitness)

        # Mutating individuals
        chromosomes = collect(map(x -> rand() < gaS.mutationRate ? gaS.mutationAlg(numberOfPoints, x, actualMinDists, partFitF) : x, chromosomes))

        # Recalculating fitness for new individuals
        fitness = collect(map(x -> maximum(fitF(x, actualMinDists)), chromosomes))

        if runS.logging != ""
            for (chr, fit) in zip(chromosomes, fitness)
                for node in chr
                    stats[node, i] += 1
                    avgFit[node, i] += fit
                end
            end
        end

        # Sorting fitness scores
        fitnessSorted = sortperm(fitness, rev=true)

        fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
        fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


        # Choosing the elit
        elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
        elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
        elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

        #=
        # Choosing the rest randomly from the others
        restNumber = gaS.populationSize - elitNumber
        restIds = [rand(fitnessSorted[elitNumber+1:end]) for _ = 1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)

        =#

        # Choosing the rest with diversity in mind 
        restNumber = gaS.populationSize - elitNumber
        avgDifferentNodes = [
            mean(
                [length(chromosomes[i]) - count(chromosomes[i] .∈ j) for j in elitChromosomes]
            )
            for i in fitnessSorted[elitNumber+1:end]
        ]
        sortedAvgDiff = sortperm(avgDifferentNodes, rev=true)
        restIds = fitnessSorted[elitNumber.+sortedAvgDiff][1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)


        chromosomes = vcat(elitChromosomes, restChromosomes)
        fitness = vcat(elitFitness, restFitness)


        if !isPerturbed && fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
        end

        if runS.logging != ""
            logRow = stats[:, i][:]
            append!(logRow, avgFit[:, i][:] ./ stats[:, i][:])
            append!(logRow, [i, maxVal])
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
    end


    maxVec, maxVal, logs
end


function maxmindp_genetic_perturbation_best(
    runS::RunSettings,
    gaS::GeneticSettings;
    fitF=DispersionProblems.calculateMinSumdp,
    partFitF=DispersionProblems.calculateSumdp,
    turnLength=200
)
    # Initializing values and functions for later use
    n, _ = size(runS.minDists)
    stats = zeros((n, runS.numberOfIterations))
    avgFit = zeros((n, runS.numberOfIterations))
    numberOfPoints, _ = size(runS.minDists)

    chromosomes = gaS.initAlg(gaS.populationSize, runS.k, runS.minDists)

    actualMinDists = runS.minDists

    # Initializing global maximum as one of the given chromosome
    maxVal = maximum(fitF(chromosomes[1], actualMinDists))
    maxVec = copy(chromosomes[1])

    # Initializing logging
    logs = []

    fitness = collect(map(x -> maximum(fitF(x, actualMinDists)), chromosomes))
    isPerturbed = true
    for i = 1:runS.numberOfIterations
        if i % turnLength == 1
            isPerturbed = !isPerturbed
            if isPerturbed
                actualMinDists = copy(runS.minDists)
                for a in maxVec
                    for b in maxVec
                        if a != b
                            if actualMinDists[a, b] > 1
                                actualMinDists[a, b] -= rand(1:actualMinDists[a, b])
                                actualMinDists[b, a] = actualMinDists[a, b]
                            end
                        end
                    end
                end
            else
                actualMinDists = copy(runS.minDists)
            end
        end
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        newChromosomes = [
            gaS.crossoverSelAlg(chromosomes, fitness, (left, right) -> gaS.crossoverAlg(left, right, actualMinDists, partFitF)) for
            _ = 1:Int(ceil(gaS.populationSize * gaS.crossoverRate))
        ]
        newFitness = collect(map(x -> maximum(fitF(x, actualMinDists)), newChromosomes)) #TODO: FIX this and test!!!!

        # Add them to the chromosome pool
        append!(chromosomes, newChromosomes)
        append!(fitness, newFitness)

        # Mutating individuals
        chromosomes = collect(map(x -> rand() < gaS.mutationRate ? gaS.mutationAlg(numberOfPoints, x, actualMinDists, partFitF) : x, chromosomes))

        # Recalculating fitness for new individuals
        fitness = collect(map(x -> maximum(fitF(x, actualMinDists)), chromosomes))

        if runS.logging != ""
            for (chr, fit) in zip(chromosomes, fitness)
                for node in chr
                    stats[node, i] += 1
                    avgFit[node, i] += fit
                end
            end
        end

        # Sorting fitness scores
        fitnessSorted = sortperm(fitness, rev=true)

        fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
        fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


        # Choosing the elit
        elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
        elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
        elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

        #=
        # Choosing the rest randomly from the others
        restNumber = gaS.populationSize - elitNumber
        restIds = [rand(fitnessSorted[elitNumber+1:end]) for _ = 1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)

        =#

        # Choosing the rest with diversity in mind 
        restNumber = gaS.populationSize - elitNumber
        avgDifferentNodes = [
            mean(
                [length(chromosomes[i]) - count(chromosomes[i] .∈ j) for j in elitChromosomes]
            )
            for i in fitnessSorted[elitNumber+1:end]
        ]
        sortedAvgDiff = sortperm(avgDifferentNodes, rev=true)
        restIds = fitnessSorted[elitNumber.+sortedAvgDiff][1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)


        chromosomes = vcat(elitChromosomes, restChromosomes)
        fitness = vcat(elitFitness, restFitness)


        if !isPerturbed && fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
        end

        if runS.logging != ""
            logRow = stats[:, i][:]
            append!(logRow, avgFit[:, i][:] ./ stats[:, i][:])
            append!(logRow, [i, maxVal])
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
    end


    maxVec, maxVal, logs
end



end