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
	using Evolutionary
end

function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

begin
	g_utils_jl = ingredients("graph_utils.jl")  
	import .g_utils_jl: WELFormat, loadgraph, loadgraphs, read_edge_list_weighted, calculate_mindist
	greedy_jl = ingredients("mmdp_greedy.jl") 
	import .greedy_jl: maxmindp_greedy_mindp
	own_genetic_jl = ingredients("mmdp_own_genetic.jl")
	import .own_genetic_jl: maxmindp_genetic_dist, maxmindp_genetic_dist2, maxmindp_genetic_dist3, maxmindp_genetic_dist4, GeneticSettings, RunSettings
	evolutionary_jl = ingredients("mmdp_evolutionary.jl")
	import .evolutionary_jl: mmdp_evolutionary, mmdp_evolutionary2 
end

# ╔═╡ f0232d89-8acf-4222-bcd4-b35d28e3d42b
begin
	files = readdir("./mmdp_graphs")
	files = collect(filter(x -> x[end-3:end] == ".dat", files))
	#files = files[1:4]
end

# ╔═╡ f60fb029-fd50-4746-80d4-e7a0241b8334
begin
	number_of_runs = 30
	greedy = false
	evolutionary = false
	memetic = false
	own_gen = true
end

# ╔═╡ c3fd688d-63cf-4cd0-9c98-c4c6e1ce56c6
df = DataFrame(graphs = files, 
	mean=zeros(length(files)),
	std=zeros(length(files)),
	min=zeros(length(files)),
	max=zeros(length(files)),
	greedy=zeros(length(files)),
	)

# ╔═╡ 2ebfb4a0-34b8-4faa-bffb-2ac8ed0a1968
function Evolutionary.trace!(record::Dict{String,Any}, objfun, state, population, method::CMAES, options)
    record["pop"] = population
end

# ╔═╡ 8bed605d-4cff-4647-ae9e-e52d38925b1c
for (i, f) in enumerate(files)
	g = loadgraph("mmdp_graphs/$(f)", WELFormat(" "))
	m_location = findfirst(x-> x == 'm', f)
	dot_location = findlast(x-> x == '.', f)
	m = parse(Int64, f[m_location + 1:dot_location-1])
	greedy_result = zeros(1:m)
	if greedy || memetic
		greedy_result = maxmindp_greedy_mindp(nv(g), m, g.weights)
		df[i, "greedy"] = calculate_mindist(greedy_result, g.weights)
	end
	if evolutionary && memetic
		results = Folds.map(_ -> mmdp_evolutionary2(nv(g), m, g.weights, greedy_result), 1:number_of_runs)
		@show Evolutionary.trace(results[1])
	elseif evolutionary
		results = Folds.map(_ -> mmdp_evolutionary(nv(g), m, g.weights), 1:number_of_runs)
	elseif own_gen && memetic
		people = [i < 11 ? copy(greedy_result) : randperm(nv(g))[1:m] for i in 1:50]
		results = Folds.map(_ -> maxmindp_genetic_dist3(nv(g), g.weights, m, 200, length(people) * 2, 0.1, 0.7, deepcopy(people)), 1:number_of_runs)
	elseif own_gen
		runS = RunSettings(g.weights, m, 200)
		gaS = GeneticSettings(200, 0.1, 0.7, 0.5)
		chromosomes = [randperm(nv(g))[1:m] for _ in 1:gaS.populationSize]
		results = Folds.map(_ -> maxmindp_genetic_dist4(runS, gaS, chromosomes), 1:number_of_runs)
	end
	values = [0.0, 0.0, 0.0, 0.0]
	if evolutionary
		values = Folds.map(x -> calculate_mindist(x.minimizer[1:m], g.weights), results)
	elseif own_gen
		values = Folds.map(x -> calculate_mindist(x, g.weights), results)
		@show values
	end
	df[i, "max"] = maximum(values)
	df[i, "min"] = minimum(values)
	df[i, "std"] = std(values)
	df[i, "mean"] = mean(values)
	if df[1,6] != 0 || df[1,2] != 0
		CSV.write("results_$(Dates.today()).csv", df)
	end
end

# ╔═╡ a29138b1-741f-4248-9e92-41736ce57d00
df

# ╔═╡ a9b1c20b-4596-4b35-923d-38a6f7e31b30
1

