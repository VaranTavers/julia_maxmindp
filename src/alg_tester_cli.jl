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

include("graph_utils.jl");
include("mmdp_greedy.jl")
include("mmdp_own_genetic.jl")

begin
	files = readdir("./mmdp_graphs")
	files = collect(filter(x -> x[end-3:end] == ".dat", files))
	#files = files[1:4]
end

# ╔═╡ f60fb029-fd50-4746-80d4-e7a0241b8334
begin
	number_of_runs = 30
	memetic = false
	#                     n_p, mut, cro, elit
	gaS = GeneticSettings(200, 0.1, 0.7, 0.5)
	number_of_iterations = 200
end

df = DataFrame(graphs = files, 
	mean=zeros(length(files)),
	std=zeros(length(files)),
	min=zeros(length(files)),
	max=zeros(length(files)),
	greedy=zeros(length(files)),
	)

for (i, f) in enumerate(files)
	g = loadgraph("mmdp_graphs/$(f)", WELFormat(" "))
	m_location = findfirst(x-> x == 'm', f)
	dot_location = findlast(x-> x == '.', f)
	m = parse(Int64, f[m_location + 1:dot_location-1])
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

	runS = RunSettings(g.weights, m, number_of_iterations)
	results = Folds.map(_ -> maxmindp_genetic_dist4(runS, gaS, chromosomes), 1:number_of_runs)
	
		values = Folds.map(x -> calculate_mindist(x, g.weights), results)
	CSV.write("run_result_$(f)_$(Dates.today()).csv", DataFrame(results, :auto))
	df[i, "max"] = maximum(values)
	df[i, "min"] = minimum(values)
	df[i, "std"] = std(values)
	df[i, "mean"] = mean(values)
	if df[1,6] != 0 || df[1,2] != 0
		CSV.write("results_compiled_$(Dates.today()).csv", df)
	end
end