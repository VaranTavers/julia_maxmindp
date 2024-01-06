begin
    using Base.Iterators
    using Statistics
end

include("../utils/graph_utils.jl")

function vector_with(v, x)
    vv = copy(v)
    push!(vv, x)

    vv
end

function maxmindp_greedy_mindp(n, k, min_dists)
    furthest = argmax(mean(min_dists, dims = 2)[:])
    points = zeros(Int64, k)
    points[1] = furthest

    for i = 2:k
        mindps =
            map(x -> calculate_mindist(vector_with(points[1:(i-1)], x), min_dists), 1:n)
        mindps[1:n.∈Ref(points[1:i-1])] .= 0
        furthest = argmax(mindps)
        points[i] = furthest
    end

    points
end
