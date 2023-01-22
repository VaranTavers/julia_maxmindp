begin
	using Base.Iterators
	using GraphPlot
	using Statistics
end

function calculate_mindist(vertices, min_distances)
	
	dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)

	minimum(map(sum,dist_sums))
end


# ╔═╡ 465d25c0-1b13-41f6-9a7a-8ee419092d83
function vector_with(v, x)
	vv = copy(v)
	push!(vv, x)

	vv
end

# ╔═╡ dc4a38c8-a636-40f2-aa9b-431a5d905e16
function maxmindp_greedy_mindp(n, k, min_dists)
	furthest = argmax(mean(min_dists, dims=2)[:])
	points = zeros(Int64, k)
	points[1] = furthest

	for i in 2:k
		mindps = map(x -> calculate_mindist(vector_with(points[1:(i-1)], x), min_dists), 1:n)
		mindps[1:n .∈ Ref(points[1:i-1])] .= 0
		furthest = argmax(mindps)
		points[i] = furthest
	end

	points
end