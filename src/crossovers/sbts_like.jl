function SBTSLikeCrossoverOne(left, right, minDists, partFitF=DispersionProblems.calculateSumdp)

    scoresLeft = collect(map(x -> partFitF(x, left, minDists), left))
    scoresRight = collect(map(x -> partFitF(x, right, minDists), right))
    leftIdsSorted = sortperm(scoresLeft, rev=true)
    rightIdsSorted = sortperm(scoresRight, rev=true)

    chromosomeLength = length(left)
    newChromosome = Set([])

    i = 1
    j = 1
    k = length(newChromosome)
    while k < chromosomeLength
        if ((i + j) % 2 == 0 && i <= chromosomeLength) || j > chromosomeLength
            push!(newChromosome, left[leftIdsSorted[i]])
            i += 1
        else
            push!(newChromosome, right[rightIdsSorted[j]])
            j += 1
        end
        k = length(newChromosome)
    end

    collect(newChromosome)
end