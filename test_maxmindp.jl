using Pkg

Pkg.activate(".")

using DispersionProblems

using Graphs
using DataFrames
using CSV
using Statistics
using Dates
using Folds
using GraphIO
using SimpleWeightedGraphs
using WeightedEdgeListFormat
using Random

include("config/config_mmdp.jl")

fst((x, _)) = x
snd((_, y)) = y

function toString(x)
    "$(x)"
end

function create_df_from_values(results, graph_names)
    local_result_df = DataFrame()

    local_result_df[!, :graphs] = graph_names

    @show results
    for (i, resultCol) in enumerate(eachcol(results))
        local_result_df[!, :x] = resultCol
        rename!(local_result_df, :x => "run$(i)")
    end

    tmpMat = Matrix(local_result_df[:, 2:(numberOfRuns+1)])
    means = mean(tmpMat, dims=2)
    stds = std(tmpMat, dims=2)
    mins = minimum(tmpMat, dims=2)
    maxs = maximum(tmpMat, dims=2)


    local_result_df[!, :mean] = means[:]
    local_result_df[!, :std] = stds[:]
    local_result_df[!, :min] = mins[:]
    local_result_df[!, :max] = maxs[:]

    local_result_df
end


configurations_GA = [
    # conf_name,                        n_p, mut, cro, elit, crossoverAlg, mutationAlg,       meme,  log,  iter 
    # Baselines
    (
        "$(gen_alg_name)_$(crossover_name)_$(crossover_selection_name)_$(mut_op_name)_$(init_alg_name)",
        DispersionProblems.Genetic.GeneticSettings(n_p, mut, cro, elit, crossover_alg, crossover_selection, mut_alg, init_alg),
        nr_gen,
        gen_alg,
    ) for n_p in ga_param_tuning_n_p, mut in ga_param_tuning_mut_rate,
    cro in ga_param_tuning_cro_rate, elit in ga_param_tuning_elit,
    (mut_op_name, mut_alg) in ga_param_tuning_mutation, nr_gen in ga_param_tuning_nr_gen,
    (gen_alg_name, gen_alg) in ga_param_tuning_genetic_alg,
    (crossover_name, crossover_alg) in ga_param_tuning_crossover,
    (crossover_selection_name, crossover_selection) in ga_param_tuning_crossover_selection,
    (init_alg_name, init_alg) in ga_param_tuning_init
]

configurations_ACO = [
    # conf_name,                        n_p, mut, cro, elit, crossoverAlg, mutationAlg,       meme,  log,  iter 
    # Baselines
    ("ACO", DispersionProblems.ACO.ACOSettings(α, β, nr_ants, ρ, ϵ, s_pheromone)) for
    α in aco_param_tuning_α, β in aco_param_tuning_β, nr_ants in aco_param_tuning_nr_ants,
    ρ in aco_param_tuning_ρ, ϵ in aco_param_tuning_ϵ,
    s_pheromone in aco_param_tuning_starting_pheromone
]



println("Alg_tester_cli started on $(Dates.now())")

if !isdir("./graphs")
    println("The graphs directory is missing.")
    exit(1)
end

graph_files = readdir("./graphs")

# Creating the necessary folders

if !isdir("./logs")
    println("Creating ./logs directory")
    mkdir("./logs")
end
if !isdir("./results")
    println("Creating ./results directory")
    mkdir("./results")
end

date_of_start = Dates.today()

version_on_date = 1

for (fitFName, fitF, partFitF) in fitFs
    @show fitFName


    while isdir("results/MDP_$(fitFName)_$(date_of_start)_$(version_on_date)")
        println("Folder results/MDP_$(fitFName)_$(date_of_start)_$(version_on_date) already exists, moving on to next number")
        global version_on_date += 1
    end

    res_folder_name = "MDP_$(fitFName)_$(date_of_start)_$(version_on_date)"
    if !isdir("logs/$(res_folder_name)")
        mkdir("logs/$(res_folder_name)")
    end
    if !isdir("results/$(res_folder_name)")
        mkdir("results/$(res_folder_name)")
    end

    # Reading graphs

    graphs = []

    for (i, graph_file) in enumerate(graph_files)

        m_location = findfirst(x -> x == 'm', graph_file)
        dot_location = findlast(x -> x == '.', graph_file)
        m = parse(Int64, graph_file[m_location+1:dot_location-1])

        push!(graphs, (graph_file, loadgraph("./graphs/$(graph_file)", WELFormat(' ')), m))
    end

    conf_num = 1

    # Exact (very slow, but ignores numberOfRuns)
    if run_exact
        println("Exact run started on $(Dates.now())")
        full_folder_name = "$(res_folder_name)/$(conf_num)_exact"

        if !isdir("logs/$(full_folder_name)")
            mkdir("logs/$(full_folder_name)")
        end
        if !isdir("results/$(full_folder_name)")
            mkdir("results/$(full_folder_name)")
        end

        open("results/$(full_folder_name)/params.txt", "w") do io
            println(io, "Exact")
        end

        values = zeros((length(graphs), numberOfRuns))
        @show values
        for (i, (graph_name, g, m)) in enumerate(graphs)
            minDists = weights(g)
            result, fitness = DispersionProblems.Exact.exact(minDists, m; fitF=fitF)

            results = [deepcopy(result) for _ in 1:numberOfRuns]
            values[i, :] .= fitness

            CSV.write(
                "results/$(full_folder_name)/run_result_$(graph_name).csv",
                DataFrame(results, :auto),
            )

            CSV.write("results/$(full_folder_name)/results_compiled.csv", create_df_from_values(values, graph_files))
        end

        global conf_num += 1
    end

    # Greedy
    if run_greedy
        println("Greedy run started on $(Dates.now())")
        full_folder_name = "$(res_folder_name)/$(conf_num)_greedy"

        if !isdir("logs/$(full_folder_name)")
            mkdir("logs/$(full_folder_name)")
        end
        if !isdir("results/$(full_folder_name)")
            mkdir("results/$(full_folder_name)")
        end

        open("results/$(full_folder_name)/params.txt", "w") do io
            println(io, "Greedy")
        end

        values = zeros((length(graphs), numberOfRuns))

        for (i, (graph_name, g, m)) in enumerate(graphs)
            minDists = weights(g)
            n = nv(g)
            result, fitness = DispersionProblems.Greedy.greedy(n, m, minDists; fitF=fitF)

            values[i, :] .= fitness
            results = [deepcopy(result) for _ in 1:numberOfRuns]

            CSV.write(
                "results/$(full_folder_name)/run_result_$(graph_name).csv",
                DataFrame(results, :auto),
            )

            CSV.write("results/$(full_folder_name)/results_compiled.csv", create_df_from_values(values, graph_files))
        end

        global conf_num += 1
    end


    # Random
    if run_random
        println("Random run started on $(Dates.now())")
        full_folder_name = "$(res_folder_name)/$(conf_num)_random"

        if !isdir("logs/$(full_folder_name)")
            mkdir("logs/$(full_folder_name)")
        end
        if !isdir("results/$(full_folder_name)")
            mkdir("results/$(full_folder_name)")
        end

        open("results/$(full_folder_name)/params.txt", "w") do io
            println(io, "Random")
        end

        values = zeros((length(graphs), numberOfRuns))

        for (i, (graph_name, g, m)) in enumerate(graphs)
            minDists = weights(g)
            n = nv(g)
            result, fitness = DispersionProblems.RandomBest.randomBest(minDists, m, random_tries, fitF)

            values[i, :] .= fitness
            results = [deepcopy(result) for _ in 1:numberOfRuns]

            CSV.write(
                "results/$(full_folder_name)/run_result_$(graph_name).csv",
                DataFrame(results, :auto),
            )

            CSV.write("results/$(full_folder_name)/results_compiled.csv", create_df_from_values(values, graph_files))
        end

        global conf_num += 1
    end


    # ACO configurations
    for (conf_name, acoS) in configurations_ACO
        # Preparing folders
        println("$(conf_name) started on $(Dates.now())")

        full_folder_name = "$(res_folder_name)/$(conf_num)_$(conf_name)"

        if !isdir("logs/$(full_folder_name)")
            mkdir("logs/$(full_folder_name)")
        end
        if !isdir("results/$(full_folder_name)")
            mkdir("results/$(full_folder_name)")
        end

        open("results/$(full_folder_name)/params.txt", "w") do io
            println(io, "acoS=", acoS)
            println(io, "runs=", aco_number_of_evaluations)
        end


        values = zeros((length(graphs), numberOfRuns))

        for (i, (graph_name, g, m)) in enumerate(graphs)
            minDists = weights(g)
            runS = DispersionProblems.ACO.RunSettings(m, fitF, partFitF, aco_number_of_evaluations, "")

            aco_results = []
            if useFolds
                @time aco_results = Folds.map(_ -> DispersionProblems.ACO.ACO_preprocessing(acoS, runS, minDists; use_folds=useFolds), 1:numberOfRuns)
            else
                @time aco_results = [DispersionProblems.ACO.ACO_preprocessing(acoS, runS, minDists; use_folds=useFolds) for _ in 1:numberOfRuns]
            end

            results = fst.(aco_results)
            values[i, :] = snd.(aco_results)


            CSV.write(
                "results/$(full_folder_name)/run_result_$(graph_name).csv",
                DataFrame(results, :auto),
            )

            CSV.write("results/$(full_folder_name)/results_compiled.csv", create_df_from_values(values, graph_files))
        end

        global conf_num += 1
    end


    # GA configurations
    for (conf_name, gaS, numberOfIterations, gen_alg) in configurations_GA
        # Preparing folders
        println("$(conf_name) started on $(Dates.now())")

        full_folder_name = "$(res_folder_name)/$(conf_num)_$(conf_name)"

        if !isdir("logs/$(full_folder_name)")
            mkdir("logs/$(full_folder_name)")
        end
        if !isdir("results/$(full_folder_name)")
            mkdir("results/$(full_folder_name)")
        end

        open("results/$(full_folder_name)/params.txt", "w") do io
            println(io, "gaS=", gaS)
        end


        values = zeros((length(graphs), numberOfRuns))

        for (i, (graph_name, g, m)) in enumerate(graphs)
            minDists = weights(g)
            runS = DispersionProblems.Genetic.RunSettings(minDists, m, numberOfIterations, false)

            ga_results = []
            if useFolds
                @time ga_results = Folds.map(_ -> gen_alg(runS, gaS; fitF=fitF), 1:numberOfRuns)
            else
                @time ga_results = [gen_alg(runS, gaS; fitF=fitF) for _ in 1:numberOfRuns]
            end

            results = fst.(ga_results)
            values[i, :] = snd.(ga_results)

            logs = map(((_, _, z),) -> z, ga_results)
            for (i, l) in enumerate(logs)
                CSV.write(
                    "logs/$(full_folder_name)/run_logs_$(graph_name)_$(i).csv",
                    DataFrame(l, :auto),
                )
            end

            CSV.write(
                "results/$(full_folder_name)/run_result_$(graph_name).csv",
                DataFrame(results, :auto),
            )

            CSV.write("results/$(full_folder_name)/results_compiled.csv", create_df_from_values(values, graph_files))
        end

        global conf_num += 1
    end

end
println("Alg_tester_cli ended on $(Dates.now())")

