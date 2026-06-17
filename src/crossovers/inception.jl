function inceptionCrossover(left, right, minDists, partFitF=DispersionProblems.calculateSumdp, algF=(_, _) -> collect(1:n))

    center = Set(left)
    for p in right
        push!(center, p)
    end

    center = collect(center)
    l_c = length(center)

    minDistsNew = [minDists[i, j] for i in 1:l_c, j in 1:l_c]

    points, _ = algF(minDistsNew, length(left))

    collect(map(x -> center[x], points))
end
