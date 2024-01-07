begin
    using Graphs
    using SimpleWeightedGraphs
    using Folds
    using Base.Iterators
    using Random
    using .GeneticCommon
end

include("../utils/graph_utils.jl")
include("./genetic_common.jl")


function copy_into!(data::Vector{Vector{Int64}}, from::Vector{Vector{Int64}}; start = 1)
    for (i, a) in enumerate(from)
        for (j, v) in enumerate(a)
            data[start+i-1][j] = v
        end
    end
end

function maxmindp_genetic_optim(
    runS::RunSettings,
    gaS::GeneticSettings,
    chromosomes_orig::Vector{Vector{Int64}},
)

    # Initializing values and functions for later use
    n = length(chromosomes_orig)
    data1 = [
        [0 for _ = 1:runS.k] for
        _ = 1:(length(chromosomes_orig)+Int(ceil(n * gaS.crossoverRate)))
    ]
    data2 = [
        [0 for _ = 1:runS.k] for
        _ = 1:(length(chromosomes_orig)+Int(ceil(n * gaS.crossoverRate)))
    ]
    fitness1 = [0.0 for _ = 1:(length(chromosomes_orig)+Int(ceil(n * gaS.crossoverRate)))]
    fitness2 = [0.0 for _ = 1:(length(chromosomes_orig)+Int(ceil(n * gaS.crossoverRate)))]


    numberOfPoints, _ = size(runS.minDists)
    calcFitness(x) = calculate_mindist(x, runS.minDists)
    runMutation(x) =
        rand() < gaS.mutationRate ? gaS.mutationAlg(numberOfPoints, x, runS.minDists) : x

    copy_into!(data1, chromosomes_orig)

    chromosomes = data1
    chromosomes_new = data2
    fitness = fitness1
    fitness_new = fitness2

    # Initializing global maximum as one of the given chromosome
    maxVal = calculate_mindist(chromosomes[1], runS.minDists)
    maxVec = copy(chromosomes[1])

    # Initializing logging
    logs = []

    for (i, c) in enumerate(chromosomes_orig)
        fitness[i] = calcFitness(c)
    end
    #@show fitness

    for i = 1:runS.numberOfIterations
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        for i = 1:Int(ceil(n * gaS.crossoverRate))
            chromosomes[n+i] =
                gaS.crossoverAlg(chromosomes[1:n], fitness[1:n], runS.minDists)
            fitness[n+i] = 1
        end


        #@show chromosomes
        # Mutating individuals
        for (i, c) in enumerate(chromosomes)
            chromosomes[i] = runMutation(c)
        end

        # Recalculating fitness for new individuals
        for (i, c) in enumerate(chromosomes)
            fitness[i] = calcFitness(c)
        end

        # Sorting fitness scores
        fitnessSorted = sortperm(fitness, rev = true)

        fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
        fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


        # Choosing the elit
        elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
        copy_into!(chromosomes_new, chromosomes[fitnessSorted[1:elitNumber]])
        for (i, f) in enumerate(fitness[fitnessSorted[1:elitNumber]])
            fitness_new[i] = f
        end

        # Choosing the rest randomly from the others
        restNumber = gaS.populationSize - elitNumber
        for i = 1:restNumber
            id = rand(fitnessSorted[elitNumber+1:end])

            for j = 1:length(chromosomes[id])
                chromosomes_new[elitNumber+i][j] = chromosomes[id][j]
            end
            fitness_new[elitNumber+i] = fitness[id]
        end

        # chromosomes = vcat(elitChromosomes, restChromosomes)

        #fitness = vcat(elitFitness, restFitness)

        if fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
        end

        if runS.logging != ""
            logRow = [i, maxVal]
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
        if i % 2 == 1
            chromosomes = data2
            chromosomes_new = data1
            fitness = fitness2
            fitness_new = fitness1
        else
            chromosomes = data1
            chromosomes_new = data2
            fitness = fitness1
            fitness_new = fitness2
        end
    end

    maxVec, logs
end
