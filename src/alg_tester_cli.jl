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
end

include("graph_utils.jl")
include("mmdp_greedy.jl")
include("mmdp_own_genetic.jl")
include("mutations/sbts.jl")
include("crossovers/naive.jl")
include("crossovers/sbts_sane.jl")

begin
    files = readdir("./mmdp_graphs")
    files = collect(filter(x -> x[end-3:end] == ".dat", files))
    #files = files[1:4]
end

fst((x, _)) = x

#=
numberOfRuns = 30

param_tuning_n_p = [50, 100, 200]
param_tuning_mut_rate = [0.1, 0.2]
param_tuning_cro_rate = [0.7, 0.8]
param_tuning_elit = [0.25, 0.5]
param_tuning_nr_gen = [200, 500, 1000]
param_tuning_mutation = [
  (mutationSBTS, "Baseline"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRouletteIN), "roulette50in"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRouletteOUT), "roulette50out"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRandomIN), "random50in"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRandomOUT), "random50out"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRouletteIN(x, p=0.25)), "roulette25in"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRouletteOUT(x, p=0.25)), "roulette25out"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRandomIN(x, p=0.25)), "random25in"),
  #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRandomOUT(x, p=0.25)), "random25out")
]
param_tuning_memetic = [false, true]
=#

numberOfRuns = 10

param_tuning_n_p = [200]
param_tuning_mut_rate = [0.2]
param_tuning_cro_rate = [0.8]
param_tuning_elit = [0.5]
param_tuning_nr_gen = [1000]
param_tuning_mutation = [
    #(mutationSBTS, "Baseline"),
    #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRouletteIN), "roulette50in"),
    #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRouletteOUT), "roulette50out"),
    #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRandomIN), "random50in"),
    #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRandomOUT), "random50out"),
    #  ((a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRouletteIN(x, p=0.25)), "roulette25in"),
    #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRouletteOUT(x, p=0.25)), "roulette25out"),
    (
        (a, b, c) -> mutationSBTS(a, b, c, in_f = x -> sumdpRandomIN(x, p = 0.25)),
        "random25in",
    ),
    #  ((a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRandomOUT(x, p=0.25)), "random25out")
]
param_tuning_memetic = [false]


configurations = [
    # conf_name,                        n_p, mut, cro, elit, crossoverAlg, mutationAlg,       meme,  log,  iter 
    # Baselines
    (
        "NEWNEWX_$(alg_name)_np$(n_p)_mut$(mut)_cro$(cro)_elit$(elit)_gen$(nr_gen)_mem$(memetic)",
        GeneticSettings(n_p, mut, cro, elit, crossoverSBTSLike, mut_alg),
        memetic,
        true,
        nr_gen,
    ) for n_p in param_tuning_n_p, mut in param_tuning_mut_rate,
    cro in param_tuning_cro_rate, elit in param_tuning_elit,
    (mut_alg, alg_name) in param_tuning_mutation, nr_gen in param_tuning_nr_gen,
    memetic in param_tuning_memetic
]

df = DataFrame(
    graphs = files,
    mean = zeros(length(files)),
    std = zeros(length(files)),
    min = zeros(length(files)),
    max = zeros(length(files)),
    greedy = zeros(length(files)),
)

if !isdir("./logs")
    mkdir("./logs")
end
if !isdir("./results")
    mkdir("./results")
end

for (conf_name, gaS, memetic, logging, numberOfIterations) in configurations
    date_of_start = Dates.today()
    if !isdir("logs/$(conf_name)_$(date_of_start)")
        mkdir("logs/$(conf_name)_$(date_of_start)")
    end
    if !isdir("results/$(conf_name)_$(date_of_start)")
        mkdir("results/$(conf_name)_$(date_of_start)")
    end
    for (i, f) in enumerate(files)
        g = loadgraph("mmdp_graphs/$(f)", WELFormat(" "))
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

        runS = RunSettings(g.weights, m, numberOfIterations, logging)
        results = Folds.map(_ -> maxmindp_genetic(runS, gaS, chromosomes), 1:numberOfRuns)
        values = Folds.map(((x, y),) -> calculate_mindist(x, g.weights), results)

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
