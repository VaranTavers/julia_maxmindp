### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ a1028eb0-5dd5-11ed-3893-69ea980733df
begin
	using Graphs
	using SimpleWeightedGraphs
	using GraphIO
	using Folds
	using Base.Iterators
	using CSV
	using DataFrames
	using GraphPlot
	using Statistics
	using Random
	using Dates
	using Evolutionary
end

# ╔═╡ 5296ebb5-46c2-4d0e-9437-d1ee0ebc7f25
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

# ╔═╡ 77e37ef3-fd63-47b6-b2fb-040b3a22dd90
begin
	g_utils_jl = ingredients("graph_utils.jl") 
	import .g_utils_jl: WELFormat, loadgraph, loadgraphs, read_edge_list_weighted, calculate_mindist
	greedy_jl = ingredients("mmdp_greedy.jl") 
	import .greedy_jl: maxmindp_greedy_mindp
	own_genetic_jl = ingredients("mmdp_own_genetic.jl")
	import .own_genetic_jl: maxmindp_genetic_dist, maxmindp_genetic_dist2, maxmindp_genetic_dist3
	evolutionary_jl = ingredients("mmdp_evolutionary.jl")
	import .evolutionary_jl: mmdp_evolutionary, mmdp_evolutionary2 
	simrank_jl = ingredients("similarities.jl") 
	import .simrank_jl: simrank_u, simrank_w, adamic_adar_u, resource_allocation_index, wcn, wra, waa
end

# ╔═╡ f0232d89-8acf-4222-bcd4-b35d28e3d42b
begin
	files = readdir("./mmdp_graphs")
	files = collect(filter(x -> x[end-3:end] == ".dat", files))
end

# ╔═╡ f60fb029-fd50-4746-80d4-e7a0241b8334
begin
	number_of_runs = 10
	greedy = false
	evolutionary = false
	memetic = true
	own_gen = true
end

# ╔═╡ ec823423-0597-4b3f-b87f-d0ca328d7b38
function get_measure(g)
	deepcopy(g.weights)
	#simrank_w(g, 100, 0.9)
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
	dist_measure = get_measure(g)
	if greedy || memetic
		greedy_result = maxmindp_greedy_mindp(nv(g), m, dist_measure)
		df[i, "greedy"] = calculate_mindist(greedy_result, dist_measure)
	end
	if evolutionary && memetic
		results = Folds.map(_ -> mmdp_evolutionary2(nv(g), m, dist_measure, greedy_result), 1:number_of_runs)
		@show Evolutionary.trace(results[1])
	elseif evolutionary
		results = Folds.map(_ -> mmdp_evolutionary(nv(g), m, dist_measure), 1:number_of_runs)
	elseif own_gen && memetic
		people = [i < 3 ? copy(greedy_result) : randperm(nv(g))[1:m] for i in 1:25]
		results = Folds.map(_ -> maxmindp_genetic_dist3(nv(g), dist_measure, m, 200, length(people) * 2, 0.1, 0.7, deepcopy(people)), 1:number_of_runs)
	elseif own_gen
		results = Folds.map(_ -> maxmindp_genetic_dist(nv(g), dist_measure, m, 1000, 50, 0.1, 0.7), 1:number_of_runs)
	end
	values = [0.0, 0.0, 0.0, 0.0]
	if evolutionary
		values = Folds.map(x -> calculate_mindist(x.minimizer[1:m], dist_measure), results)
	elseif own_gen
		values = Folds.map(x -> calculate_mindist(x, dist_measure), results)
		@show values
	end
	df[i, "max"] = maximum(values)
	df[i, "min"] = minimum(values)
	df[i, "std"] = std(values)
	df[i, "mean"] = mean(values)
	if df[i,6] != 0 || df[i,2] != 0
	CSV.write("results_$(Dates.today()).csv", df)
	end
end

# ╔═╡ a29138b1-741f-4248-9e92-41736ce57d00
df

# ╔═╡ a9b1c20b-4596-4b35-923d-38a6f7e31b30
if df[1,6] != 0 || df[1,2] != 0
	CSV.write("results_$(Dates.today()).csv", df)
end

# ╔═╡ 3942cd06-5377-414e-9c20-efb8531e21ff
a = sort([80, 411, 131, 354, 366, 130, 259, 209, 143, 413, 156, 427, 356, 165, 239, 90, 92, 214, 137, 291, 283, 61, 407, 227, 240, 163, 369, 79, 169, 484, 400, 447, 84, 83, 264, 274, 26, 293, 56, 305, 9, 315, 414, 159, 403, 303, 28, 351, 459, 247, 360, 6, 441, 114, 162, 242, 51, 232, 334, 44, 479, 321, 433, 464, 30, 379, 168, 490, 387, 85, 443, 133, 449, 66, 422, 190, 395, 171, 424, 160, 110, 236, 267, 15, 72, 352, 412, 250, 491, 335, 222, 340, 428, 39, 367, 482, 106, 394, 376, 270, 111, 101, 316, 299, 255, 42, 276, 486, 390, 409, 489, 140, 330, 37, 207, 477, 418, 117, 300, 16, 348, 243, 235, 266, 113, 238, 421, 493, 35, 224, 53, 423, 465, 435, 265, 273, 120, 345, 362, 476, 281, 286, 221, 292, 319, 48, 426, 350, 157, 108, 297, 213, 178, 375, 197, 24, 466, 280, 480, 488, 434, 338, 481, 420, 97, 440, 363, 155, 398, 461, 99, 38, 269, 233, 333, 469, 152, 201, 223, 33, 364, 453, 212, 136, 408, 452, 438, 381, 2, 55, 337, 23, 302, 206, 343, 336, 103, 324, 355, 377])

# ╔═╡ e613153c-cff4-44dc-8c59-77a8e241c9ee
sbts[findlast(a .!== sbts)]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
Evolutionary = "86b6b26d-c046-49b6-aa0b-5f0f74682bd6"
Folds = "41a02a25-b8f0-4f67-bc48-60067656b558"
GraphIO = "aa1b3936-2fda-51b9-ab35-c553d3a640a2"
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SimpleWeightedGraphs = "47aef6b3-ad0c-573a-a1e2-d07658019622"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.10.8"
DataFrames = "~1.4.4"
Evolutionary = "~0.11.1"
Folds = "~0.2.8"
GraphIO = "~0.6.0"
GraphPlot = "~0.5.2"
Graphs = "~1.7.4"
SimpleWeightedGraphs = "~1.2.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "7c9f23876c24438dfcb17d4d8aa71ab85dbe0807"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "3fa8cc751763c91a5ea33331e523221009cb1e6f"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.23"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "0310e08cb19f5da31d08341c6120c047598f5b9c"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.5.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "14c3f84a763848906ac681f94cf469a851601d92"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.28"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "00a2cccc7f098ff3b66806862d275ca3db9e6e5a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.5.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "d853e57661ba3a57abcdaa201f4c9917a93487a2"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "fb21ddd70a051d882a1686a5a550990bbe371a95"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d4f69885afa5e6149d0cab3818491565cf41446d"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.4.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "c5b6685d53f933c11404a3ae9822afe30d522494"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.12.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Evolutionary]]
deps = ["LinearAlgebra", "NLSolversBase", "Random", "StackViews", "Statistics", "UnPack"]
git-tree-sha1 = "495cc997e315ee71d2361bf4c90ef21dc12e8fd1"
uuid = "86b6b26d-c046-49b6-aa0b-5f0f74682bd6"
version = "0.11.1"

[[deps.ExternalDocstrings]]
git-tree-sha1 = "1224740fc4d07c989949e1c1b508ebd49a65a5f6"
uuid = "e189563c-0753-4f5e-ad5c-be4293c83fb4"
version = "0.1.1"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "04ed1f0029b6b3af88343e439b995141cb0d0b8d"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.17.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Folds]]
deps = ["Accessors", "BangBang", "Baselet", "DefineSingletons", "Distributed", "ExternalDocstrings", "InitialValues", "MicroCollections", "Referenceables", "Requires", "Test", "ThreadedScans", "Transducers"]
git-tree-sha1 = "638109532de382a1f99b1aae1ca8b5d08515d85a"
uuid = "41a02a25-b8f0-4f67-bc48-60067656b558"
version = "0.2.8"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "a69dd6db8a809f78846ff259298678f0d6212180"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.34"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GraphIO]]
deps = ["DelimitedFiles", "Graphs", "Requires", "SimpleTraits"]
git-tree-sha1 = "c243b56234de8afbb6838129e72a4dfccd230ccc"
uuid = "aa1b3936-2fda-51b9-ab35-c553d3a640a2"
version = "0.6.0"

[[deps.GraphPlot]]
deps = ["ArnoldiMethod", "ColorTypes", "Colors", "Compose", "DelimitedFiles", "Graphs", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5cd479730a0cb01f880eff119e9803c13f214cab"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.5.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "0cf92ec945125946352f3d46c96976ab972bde6f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.3.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.InvertedIndices]]
git-tree-sha1 = "82aec7a3dd64f4d9584659dc0b62ef7db2ef3e19"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.2.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "4d5917a26ca33c66c8e5ca3247bd163624d35493"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "f71d8950b724e9ff6110fc948dff5a329f901d64"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.8"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "8175fc2b118a3755113c8e68084dc1a9e63c61ee"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "96f6db03ab535bdb901300f88335257b0018689d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Referenceables]]
deps = ["Adapt"]
git-tree-sha1 = "e681d3bfa49cd46c3c161505caddf20f0e62aaa9"
uuid = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"
version = "0.1.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "c02bd3c9c3fc8463d3591a62a378f90d2d8ab0f3"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.17"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "6954a456979f23d05085727adb17c4551c19ecd1"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.12"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ThreadedScans]]
deps = ["ArgCheck"]
git-tree-sha1 = "ca1ba3000289eacba571aaa4efcefb642e7a1de6"
uuid = "24d252fe-5d94-4a69-83ea-56a14333d47a"
version = "0.1.0"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c42fa452a60f022e9e087823b47e5a5f8adc53d5"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.75"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═a1028eb0-5dd5-11ed-3893-69ea980733df
# ╠═5296ebb5-46c2-4d0e-9437-d1ee0ebc7f25
# ╠═77e37ef3-fd63-47b6-b2fb-040b3a22dd90
# ╠═f0232d89-8acf-4222-bcd4-b35d28e3d42b
# ╠═f60fb029-fd50-4746-80d4-e7a0241b8334
# ╠═ec823423-0597-4b3f-b87f-d0ca328d7b38
# ╠═c3fd688d-63cf-4cd0-9c98-c4c6e1ce56c6
# ╠═2ebfb4a0-34b8-4faa-bffb-2ac8ed0a1968
# ╠═8bed605d-4cff-4647-ae9e-e52d38925b1c
# ╠═a29138b1-741f-4248-9e92-41736ce57d00
# ╠═a9b1c20b-4596-4b35-923d-38a6f7e31b30
# ╠═3942cd06-5377-414e-9c20-efb8531e21ff
# ╠═e613153c-cff4-44dc-8c59-77a8e241c9ee
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
