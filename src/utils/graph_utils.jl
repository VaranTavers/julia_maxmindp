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

#=function calculate_mindist(vertices, min_distances)

    dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)
    # TODO: Change back inner minimum -> sum!!!
    sum(map(sum, dist_sums))
end=#


function calculate_mindist(vertices, min_distances)
    dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)
    minimum(map(sum, dist_sums))
end


"""
    Calculates distasnces of the graph.
"""
function generate_distances_mat(g)
    res = zeros((nv(g), nv(g)))

    for i in 1:nv(g)
        shortest_path = dijkstra_shortest_paths(g, i)
        res[i, :] = collect(map(x -> x != Inf ? x : 0, shortest_path.dists))
    end


    res
end