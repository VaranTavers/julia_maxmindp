using Graphs
using SimpleWeightedGraphs
using Random

function maxmindp_random(g, k)
	randperm(nv(g))[1:k]
end