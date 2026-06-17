module Greedy
using Base.Iterators
using Statistics
using ..DispersionProblems
using ..DispersionProblems.GraphUtils

function vector_with(v, x)
    vv = copy(v)
    push!(vv, x)

    vv
end

function greedy(n, m, minDists; fitF=calculateMinSumdp)
    furthest = argmax(mean(minDists, dims=2)[:])
    points = zeros(Int64, m)
    points[1] = furthest

    for i = 2:m
        mindps =
            map(x -> x ∈ points ? 0 : fitF(vector_with(points[1:(i-1)], x), minDists), 1:n)
        furthest = argmax(mindps)
        points[i] = furthest
    end

    points, maximum(fitF(points, minDists))
end
end