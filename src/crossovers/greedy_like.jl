function vector_with(v, x)
    vv = copy(v)
    push!(vv, x)

    vv
end

function greedyLikeCrossover(left, right, minDists, partFitF=DispersionProblems.calculateSumdp, fitF=DispersionProblems.calculateMinSumdp)

    center = Set(left)
    for p in right
        push!(center, p)
    end

    center = collect(center)
    l_c = length(center)
    m = length(left)

    minDistsNew = [minDists[i, j] for i in 1:l_c, j in 1:l_c]

    starter = rand(1:l_c)
    points = zeros(Int64, m)
    points[1] = starter

    mindps = collect(map(x -> x == starter ? 0 : minDistsNew[x, starter], 1:l_c))
    for i = 2:m
        furthest = argmax(mindps)
        points[i] = furthest
        mindps += minDistsNew[furthest, :]
        mindps[points[1:i]] .= 0
    end

    collect(map(x -> center[x], points))
end
