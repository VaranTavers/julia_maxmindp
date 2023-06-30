using Graphs
using SimpleWeightedGraphs
using Random

include("graph_utils.jl")

function maxmindp_random(n, k, min_dists)
	randperm(n)[1:k]
end

function maxmindp_bo_random(n, k, min_dists, m)
	runs = [maxmindp_random(n, k, min_dists) for _ in 1:m]
	results = map(x -> calculate_mindist(x, min_dists), runs)

	runs[argmax(results)]
end