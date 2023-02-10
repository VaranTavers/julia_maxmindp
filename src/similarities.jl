begin
	using Graphs
	using SimpleWeightedGraphs
	using Base.Iterators
end

include("graph_utils.jl") 

mutable struct SimRankInner
	decay_factor::Float64
	old_sim
	node_num::Int64
	new_sim
end

function init_sim(g, decay_factor)
	n = nv(g)
	SimRankInner(
		decay_factor,
		[i == j ? 1 : 0 for i in 1:n, j in 1:n],
		n,
		zeros(n, n)
	)
end

function calculate_simrank(g, inner::SimRankInner, node1, node2)
	if node1 == node2
		return 1
	end

	in_neighbors1 = inneighbors(g, node1)
	in_neighbors2 = inneighbors(g, node2)

	if length(in_neighbors1) == 0 || length(in_neighbors2) == 0
		return 0
	end

	simrank_sum = sum(
		map(((x,y),) -> inner.old_sim[x, y],
		Iterators.product(in_neighbors1, in_neighbors2)
		)
	)

	scale = inner.decay_factor / (length(in_neighbors1) * length(in_neighbors2))

	scale * simrank_sum
end

function simrank_u2(g, iterations, decay_factor)
	inner = init_sim(g, decay_factor)
	n = inner.node_num
	for _ in 1:iterations
		# One iteration
		for (node1, node2) in Iterators.product(1:n, 1:n)
			inner.new_sim[node1, node2] = calculate_simrank(g, inner, node1, node2)
		end
		inner.old_sim = deepcopy(inner.new_sim)
	end

	inner.old_sim
end

function simrank_u(g, iterations, decay_factor)
	n = nv(g)
	new_sim = zeros(n, n)
	for i in 1:n
		new_sim[i, i] = 1
	end
	weights = g.weights .> 0
	in_neighbors_l = [length(inneighbors(g, i)) * length(inneighbors(g, j)) for i in 1:n, j in 1:n]
	for _ in 1:iterations
		new_sim = decay_factor * (weights * new_sim * weights') ./ in_neighbors_l
		for i in 1:n
			new_sim[i, i] = 1
		end
	end

	new_sim
end

function calculate_evidence(g)
	t = g.weights .> 0
	E = t * t'
	1 .- 0.5 .^ E
end

function simrank_w(g, iterations, decay_factor)
	n = nv(g)
	new_sim = zeros(n, n)
	for i in 1:n
		new_sim[i, i] = 1
	end
	E = calculate_evidence(g)
	weights = copy(g.weights) / maximum(g.weights)
	in_neighbors_l = [length(inneighbors(g, i)) * length(inneighbors(g, j)) for i in 1:n, j in 1:n]
	for _ in 1:iterations
		new_sim = decay_factor * (weights * new_sim * weights') ./ in_neighbors_l
		for i in 1:n
			new_sim[i, i] = 1
		end
	end

	E .* new_sim
end

function inner_sum(f, neighbors, i, j)
	sum(
		[1 / f(length(neighbors[x])) for x in intersect(neighbors[i], neighbors[j])]
	)
end

function adamic_adar_u(g)
	n = nv(g)
	neighbors_list = [Set(neighbors(g, i)) for i in 1:n]
	
	[inner_sum(log, neighbors_list, i, j) for i in 1:n, j in 1:n]
end

function resource_allocation_index(g)
	n = nv(g)
	neighbors_list = [Set(neighbors(g, i)) for i in 1:n]
	
	[inner_sum(x->x, neighbors_list, i, j) for i in 1:n, j in 1:n]
end

function strength(neighbors, weights, i, α)
	sum([weights[i, j]^α for j in neighbors[i]])
end

function wcn_inner_sum(neighbors, i, j, weights, α)
	sum(
		[weights[i, x]^α + weights[x, j]^α for x in intersect(neighbors[i], neighbors[j])]
	)
end

function waa_inner_sum(neighbors, i, j, weights, α)
	sum(
		[(weights[i, x]^α + weights[x, j]^α) / log(1 + strength(neighbors, weights, x, α)) for x in intersect(neighbors[i], neighbors[j])]
	)
end

function wra_inner_sum(neighbors, i, j, weights, α)
	sum(
		[(weights[i, x]^α + weights[x, j]^α) / strength(neighbors, weights, x, α) for x in intersect(neighbors[i], neighbors[j])]
	)
end

function wcn(g; α=1)
	n = nv(g)
	neighbors_list = [Set(neighbors(g, i)) for i in 1:n]

	weights = g.weights
	
	[wcn_inner_sum(neighbors_list, i, j, weights, α) for i in 1:n, j in 1:n]
end

function waa(g; α=1)
	n = nv(g)
	neighbors_list = [Set(neighbors(g, i)) for i in 1:n]

	weights = g.weights
	
	[waa_inner_sum(neighbors_list, i, j, weights, α) for i in 1:n, j in 1:n]
end

function wra(g; α=1)
	n = nv(g)
	neighbors_list = [Set(neighbors(g, i)) for i in 1:n]

	weights = g.weights
	
	[wra_inner_sum(neighbors_list, i, j, weights, α) for i in 1:n, j in 1:n]
end


function get_g4()
	g4 = SimpleWeightedDiGraph(7)
	
	
	add_edge!(g4, 1, 2, 1.0)
	add_edge!(g4, 1, 3, 1.0)
	add_edge!(g4, 1, 4, 1.0)
	add_edge!(g4, 1, 5, 1.0)
	add_edge!(g4, 1, 7, 1.0)
	add_edge!(g4, 2, 1, 1.0)
	add_edge!(g4, 3, 1, 1.0)
	add_edge!(g4, 3, 2, 1.0)
	add_edge!(g4, 4, 2, 1.0)
	add_edge!(g4, 4, 3, 1.0)
	add_edge!(g4, 4, 5, 1.0)
	add_edge!(g4, 5, 1, 1.0)
	add_edge!(g4, 5, 3, 1.0)
	add_edge!(g4, 5, 4, 1.0)
	add_edge!(g4, 5, 6, 1.0)
	add_edge!(g4, 6, 1, 1.0)
	add_edge!(g4, 6, 5, 1.0)
	add_edge!(g4, 7, 5, 1.0)

	g4
end


function test_simrank()
	g4 = get_g4()
	simrank_u(g4, 100, 0.9), simrank_u2(g4, 100, 0.9), simrank_w(g4, 100, 0.9)
end

function test_aa()
	g4 = get_g4()
	
	adamic_adar_u(g4), resource_allocation_index(g4)
end