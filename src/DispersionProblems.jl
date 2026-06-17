module DispersionProblems

using Graphs
using SimpleWeightedGraphs
using Statistics

function calculateSumdp(newPoint, v, minDists)
    sum(map(x -> minDists[x, newPoint], filter(x -> x != newPoint, v)))
end

function calculateMindp(newPoint, v, minDists)
    minimum(map(x -> minDists[x, newPoint], filter(x -> x != newPoint, v)))
end

function calculateMaxdp(newPoint, v, minDists)
    if length(filter(x -> x != newPoint, v)) == 0
        @show v
        @show newPoint
    end
    maximum(map(x -> minDists[x, newPoint], filter(x -> x != newPoint, v)))
end

function calculateMeandp(newPoint, v, minDists)
    mean(map(x -> minDists[x, newPoint], filter(x -> x != newPoint, v)))
end

function calculateStddp(newPoint, v, minDists)
    std(map(x -> minDists[x, newPoint], filter(x -> x != newPoint, v)))
end

function calculateMinSumdp(vals, minDists)
    minimum([calculateSumdp(x, vals, minDists) for x in vals])
end

function calculateMaxSumdp(vals, minDists)
    maximum([calculateSumdp(x, vals, minDists) for x in vals])
end

function calculateMinMaxdp(vals, minDists)
    minimum([calculateMaxdp(x, vals, minDists) for x in vals])
end

function calculateMinMindp(vals, minDists)
    minimum([calculateMindp(x, vals, minDists) for x in vals])
end

function calculateMaxMaxdp(vals, minDists)
    maximum([calculateMaxdp(x, vals, minDists) for x in vals])
end

function calculateMaxMindp(vals, minDists)
    maximum([calculateMindp(x, vals, minDists) for x in vals])
end

function calculateMinMeandp(vals, minDists)
    minimum([calculateMeandp(x, vals, minDists) for x in vals])
end

function calculateMaxMeandp(vals, minDists)
    maximum([calculateMeandp(x, vals, minDists) for x in vals])
end

function calculateMinStddp(vals, minDists)
    minimum([calculateStddp(x, vals, minDists) for x in vals])
end

function calculateMaxStddp(vals, minDists)
    maximum([calculateStddp(x, vals, minDists) for x in vals])
end


function calculateSum(vals, minDists)
    sum([i >= j ? 0 : minDists[x, y] for (i, x) in enumerate(vals) for (j, y) in enumerate(vals)])
end


include("utils/graph_utils.jl")
include("algorithms/mmdp_own_genetic.jl")
include("algorithms/exact.jl")
include("algorithms/mmdp_greedy.jl")
include("algorithms/random_best.jl")
include("algorithms/aco.jl")


end
