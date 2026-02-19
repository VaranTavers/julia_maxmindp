begin
    using Graphs
    using SimpleWeightedGraphs
    using Folds
    using Base.Iterators
    using CSV
    using DataFrames
    using Statistics
    using Random
    using Dates
    using WeightedEdgeListFormat

    include("algorithms/genetic_common.jl")
    using .GeneticCommon
end

include("utils/graph_utils.jl")
include("algorithms/mmdp_greedy.jl")
include("algorithms/mmdp_own_genetic.jl")
include("algorithms/mmdp_tree_genetic.jl")
include("algorithms/mmdp_own_optim.jl")
include("mutations/sbts.jl")
include("mutations/random.jl")
include("crossovers/naive.jl")
include("crossovers/sbts_sane.jl")

begin
    files = readdir("./mmdp_graphs")
    files = collect(filter(x -> x[end-3:end] == ".dat", files))
    #files = files[1:4]
end

println("Alg_tester_cli started on $(Dates.now())")

fst((x, _)) = x

# CONFIGURATION

#=
param_tuning_n_p = [50, 100, 200]
param_tuning_mut_rate = [0.1, 0.2]
param_tuning_cro_rate = [0.7, 0.8]
param_tuning_elit = [0.25, 0.5]
param_tuning_nr_gen = [200, 500, 1000]
=#

numberOfRuns = 30

param_tuning_n_p = [200]
param_tuning_mut_rate = [0.2]
param_tuning_cro_rate = [0.7]
param_tuning_elit = [0.25]
param_tuning_nr_gen = [85000]
param_tuning_mutation = [
    ((a, b, _c) -> mutate(a, b), "PEERJ_Naive")
    ((a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRouletteIN), "PEERJ_roulette50in")
    #(mutationSBTS, "Baseline"),
    #=      ((a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRouletteOUT), "roulette50out"),
          ((a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRandomIN), "random50in"),
          ((a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRandomOUT), "random50out"),
          ((a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRouletteIN(x, p=0.25)), "roulette25in"),
          ((a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRouletteOUT(x, p=0.25)), "roulette25out"),
        (
            (a, b, c) -> mutationSBTS(a, b, c, in_f = x -> sumdpRandomIN(x, p = 0.25)),
            "random25in",
        ),
          ((a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRandomOUT(x, p=0.25)), "random25out")
    =#
]

param_tuning_genetic_alg = [
    ("GA", maxmindp_genetic),
    ("GA+", maxmindp_genetic_tree)
]
param_tuning_crossover = [
    ("PEERJ", crossoverSBTSSane)
    ("Naive", crossoverRoulette)
]
param_tuning_memetic = [false]

# END OF CONFIGURATION


configurations = [
    # conf_name,                        n_p, mut, cro, elit, crossoverAlg, mutationAlg,       meme,  log,  iter 
    # Baselines
    (
        "$(gen_alg_name)_$(crossover_name)_$(mut_op_name)_$(n_p)_$(mut)_$(cro)_$(elit)_$(nr_gen)",
        GeneticSettings(n_p, mut, cro, elit, crossover_alg, mut_alg),
        memetic,
        true,
        nr_gen,
        (gen_alg_name, gen_alg),
    ) for n_p in param_tuning_n_p, mut in param_tuning_mut_rate,
    cro in param_tuning_cro_rate, elit in param_tuning_elit,
    (mut_alg, mut_op_name) in param_tuning_mutation, nr_gen in param_tuning_nr_gen,
    memetic in param_tuning_memetic,
    (gen_alg_name, gen_alg) in param_tuning_genetic_alg,
    (crossover_name, crossover_alg) in param_tuning_crossover
]

l_files = length(files)

df = DataFrame(
    graphs=files,
    mean=zeros(l_files),
    std=zeros(l_files),
    min=zeros(l_files),
    max=zeros(l_files),
    greedy=zeros(l_files),
)

if !isdir("./logs")
    mkdir("./logs")
end
if !isdir("./results")
    mkdir("./results")
end

for (conf_name, gaS, memetic, logging, numberOfIterations, (gen_alg_name, gen_alg)) in
    configurations
    println("$(conf_name) started on $(Dates.now())")

    date_of_start = Dates.today()
    if !isdir("logs/$(conf_name)_$(date_of_start)")
        mkdir("logs/$(conf_name)_$(date_of_start)")
    end
    if !isdir("results/$(conf_name)_$(date_of_start)")
        mkdir("results/$(conf_name)_$(date_of_start)")
    end
    open("results/$(conf_name)_$(date_of_start)/params.txt", "a") do io
        println(io, "gaS=", gaS)
        println(io, "memetic=", memetic)
        println(io, "gen_alg_name=", gen_alg_name)
        println(io, "iter=", numberOfIterations)
    end
    for (i, f) in enumerate(files)
        println("$(i)/$(length(files)) $(f) started on $(Dates.now())")

        g = loadgraph("mmdp_graphs/$(f)", WELFormat(' '))
        m_location = findfirst(x -> x == 'm', f)
        dot_location = findlast(x -> x == '.', f)
        m = parse(Int64, f[m_location+1:dot_location-1])
        greedy_result = zeros(m)

        if memetic
            greedy_result = maxmindp_greedy_mindp(nv(g), m, g.weights)
            df[i, "greedy"] = calculate_mindist(greedy_result, g.weights)
        end

        if memetic
            chromosomes = [
                i < gaS.populationSize / 5 ? copy(greedy_result) : randperm(nv(g))[1:m]
                for i = 1:gaS.populationSize
            ]
        else
            chromosomes = [randperm(nv(g))[1:m] for _ = 1:gaS.populationSize]
        end

        weights = g.weights#generate_distances_mat(g)

        runS = RunSettings(weights, m, numberOfIterations, logging)
        @time results = Folds.map(_ -> gen_alg(runS, gaS, chromosomes), 1:numberOfRuns)
        values = Folds.map(((x, y),) -> calculate_mindist(x, weights), results)

        if logging
            max_val = argmax(values)
            _, logs = results[max_val]
            CSV.write(
                "logs/$(conf_name)_$(date_of_start)/logs_$(f)_$(max_val).csv",
                DataFrame(logs, :auto),
            )
        end

        CSV.write(
            "results/$(conf_name)_$(date_of_start)/run_result_$(f).csv",
            DataFrame(fst.(results), :auto),
        )
        df[i, "max"] = maximum(values)
        df[i, "min"] = minimum(values)
        df[i, "std"] = std(values)
        df[i, "mean"] = mean(values)
        if df[1, 6] != 0 || df[1, 2] != 0
            CSV.write("results/$(conf_name)_$(date_of_start)/results_compiled.csv", df)
        end
    end
end


println("Alg_tester_cli ended on $(Dates.now())")
