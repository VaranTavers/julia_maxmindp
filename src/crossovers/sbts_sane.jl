
function calculateSumdp(newPoint, v, minDists)
    sum(map(x -> minDists[x, newPoint], v))
end

function SBTSLikeCrossoverSane(left, right, minDists)

    center = Set(left)
    for p in right
        push!(center, p)
    end

    center = collect(center)
    scoresCenter = collect(map(x -> calculateSumdp(x, center, minDists), center))
    centerIdsSorted = sortperm(scoresCenter, rev=true)

    collect(map(x -> center[x], centerIdsSorted[1:length(left)]))
end

function crossoverSBTSSane(chromosomes, fitness, minDists)
    rouletteWheel = fitness ./ sum(fitness)


    SBTSLikeCrossoverSane(
        chromosomes[sample(rouletteWheel)],
        chromosomes[sample(rouletteWheel)],
        minDists,
    )
end
