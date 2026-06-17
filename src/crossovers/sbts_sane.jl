function SBTSLikeCrossoverSane(left, right, minDists, partFitF=DispersionProblems.calculateSumdp)

    center = Set(left)
    for p in right
        push!(center, p)
    end

    center = collect(center)
    scoresCenter = collect(map(x -> partFitF(x, center, minDists), center))
    centerIdsSorted = sortperm(scoresCenter, rev=true)

    collect(map(x -> center[x], centerIdsSorted[1:length(left)]))
end
