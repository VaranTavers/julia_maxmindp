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
include("crossovers/sbts_like.jl")

begin
  files = readdir("./mmdp_graphs")
  files = collect(filter(x -> x[end-3:end] == ".dat", files))
  #files = files[1:4]
end

fst((x, _)) = x


numberOfRuns = 30

configurations = [
  # conf_name,                        n_p, mut, cro, elit, crossoverAlg, mutationAlg,       meme,  log,  iter 
  # Baselines
  ("baseline_ga_200",
    GeneticSettings(
      200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      mutationSBTS
    ),
    false, true, 200),
  ("baseline_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      mutationSBTS,
    ),
    true, true, 200),
  # GA variations 50%
  ("roulette50in_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRouletteIN)
    ),
    false, true, 200),
  ("roulette50out_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRouletteOUT),
    ),
    false, true, 200),
  ("random50in_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRandomIN)
    ),
    false, true, 200),
  ("random50out_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRandomOUT)
    ),
    false, true, 200),
  # GA variations 25%
  ("roulette25in_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRouletteIN(x, p=0.25))
    ),
    false, true, 200),
  ("roulette25out_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRouletteOUT(x, p=0.25)),
    ),
    false, true, 200),
  ("random25in_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRandomIN(x, p=0.25))
    ),
    false, true, 200),
  ("random25out_ga_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRandomOUT(x, p=0.25))
    ),
    false, true, 200),
  # Memetic variations (50%)
  ("roulette50in_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRouletteIN)
    ),
    true, true, 200),
  ("roulette50out_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRouletteOUT),
    ),
    true, true, 200),
  ("random50in_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=sumdpRandomIN)
    ),
    true, true, 200),
  ("random50out_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=sumdpRandomOUT)
    ),
    true, true, 200),
  # Memetic variations 25%
  ("roulette25in_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRouletteIN(x, p=0.25))
    ),
    true, true, 200),
  ("roulette25out_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRouletteOUT(x, p=0.25)),
    ),
    true, true, 200),
  ("random25in_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, in_f=x -> sumdpRandomIN(x, p=0.25))
    ),
    true, true, 200),
  ("random25out_memetic_200",
    GeneticSettings(200, 0.1, 0.7, 0.5,
      crossoverSBTSLike,
      (a, b, c) -> mutationSBTS(a, b, c, out_f=x -> sumdpRandomOUT(x, p=0.25))
    ),
    true, true, 200),
]

df = DataFrame(graphs=files,
  mean=zeros(length(files)),
  std=zeros(length(files)),
  min=zeros(length(files)),
  max=zeros(length(files)),
  greedy=zeros(length(files)),
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
      chromosomes = [i < gaS.populationSize / 5 ? copy(greedy_result) : randperm(nv(g))[1:m] for i in 1:gaS.populationSize]
    else
      chromosomes = [randperm(nv(g))[1:m] for _ in 1:gaS.populationSize]
    end

    runS = RunSettings(g.weights, m, numberOfIterations, logging)
    results = Folds.map(_ -> maxmindp_genetic(runS, gaS, chromosomes), 1:numberOfRuns)
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