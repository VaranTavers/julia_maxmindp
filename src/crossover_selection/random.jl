function randomSelection(chromosomes, fitness, crossAlg; forceSeparate=true)
    l = length(chromosomes)

    c1 = rand(1:l)
    c2 = rand(1:l)

    if forceSeparate
        if l == 1
            @show "Not enough population"
            return deepcopy(chromosomes[1])
        end
        while c1 == c2
            c2 = rand(1:l)
        end
    end

    crossAlg(
        chromosomes[c1],
        chromosomes[c2]
    )
end