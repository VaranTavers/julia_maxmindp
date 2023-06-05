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

begin
  files = readdir("./mmdp_graphs")
  files = collect(filter(x -> x[end-3:end] == ".dat", files))
  #files = files[1:4]
end

fst((x, _)) = x


numberOfRuns = 30

configurations = [
  # conf_name,                        n_p, mut, cro, elit, mutationAlg, crossoverAlg,       meme,  log,  iter 
  ("baseline_ga_200", GeneticSettings(200, 0.1, 0.7, 0.5, mutationSBTS, crossoverRoulette), false, true, 200)
  ("baseline_memetic_200", GeneticSettings(200, 0.1, 0.7, 0.5, mutationSBTS, crossoverRoulette), true, true, 200)
]

df = DataFrame(graphs=files,
  mean=zeros(length(files)),
  std=zeros(length(files)),
  min=zeros(length(files)),
  max=zeros(length(files)),
  greedy=zeros(length(files)),
)

for (conf_name, gaS, memetic, logging, numberOfIterations) in configurations
  date_of_start = Dates.today()
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
      chromosomes = [i < gaS.populationSize / 5 ? copy(greedy_result) : randperm(nv(g))[1:m] for i in 1:gaS.populationSize]
    else
      chromosomes = [randperm(nv(g))[1:m] for _ in 1:gaS.populationSize]
    end

    runS = RunSettings(g.weights, m, numberOfIterations, logging)
    results = Folds.map(_ -> maxmindp_genetic_dist4(runS, gaS, chromosomes), 1:numberOfRuns)
    values = Folds.map(((x, y),) -> calculate_mindist(x, g.weights), results)

    if logging
      max_val = argmax(values)
      _, logs = results[max_val]
      CSV.write("logs/$(conf_name)_$(date_of_start)/logs_$(f)_$(max_val).csv", DataFrame(logs, :auto))
    end

    CSV.write("results/$(conf_name)_$(date_of_start)/run_result_$(f).csv", DataFrame(fst.(results), :auto))
    df[i, "max"] = maximum(values)
    df[i, "min"] = minimum(values)
    df[i, "std"] = std(values)
    df[i, "mean"] = mean(values)
    if df[1, 6] != 0 || df[1, 2] != 0
      CSV.write("results/$(conf_name)_$(date_of_start)/results_compiled.csv", df)
    end
  end
end