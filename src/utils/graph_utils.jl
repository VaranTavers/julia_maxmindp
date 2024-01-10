begin
    using Graphs
    using SimpleWeightedGraphs
    using GraphIO
    using CSV
    using DataFrames
end

function getindex(elem, v)
    findfirst(x -> x == elem, v)
end

function calculate_mindist(vertices, min_distances)

    dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)

    minimum(map(sum, dist_sums))
end
