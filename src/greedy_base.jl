begin
    using Graphs
    using SimpleWeightedGraphs
    using Statistics
end

function vector_with(v, x)
	vv = copy(v)
	push!(vv, x)

	vv
end

# f1, f2 example: maxmindp -> f1=argmax, f2=minimum
function greedy_k_cent(g, cent, k, f1, f2)
	min_dists = copy(g.weights)
	
    mindps = cent(g)
	points = zeros(Int64, k)
	points[1] = f1(mindps)

	for i in 2:k
		mindps = map(x -> f2(cent(g, vector_with(points[1:(i-1)], x))), 1:nv(g))
		mindps[1:nv(g) .∈ Ref(points[1:i-1])] .= 0
		furthest = f1(mindps)
		points[i] = furthest
	end

	points
end