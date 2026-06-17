function rouletteWheelSelection(chromosomes, fitness, crossAlg; forceSeparate=true)
    if sum(fitness) == 0
        fitness .+= 1
    end

    rouletteWheel = fitness ./ sum(fitness)

    c1 = sample(rouletteWheel)
    c2 = sample(rouletteWheel)

    if forceSeparate
        i = 0
        while c1 == c2 && i < 100
            c2 = sample(rouletteWheel)
            i += 1
        end

        # If rouletteWheel fails to find two distinct values in 100 runs
        while c1 == c2
            c2 = rand(1:length(rouletteWheel))
        end
    end

    crossAlg(
        chromosomes[c1],
        chromosomes[c2]
    )
end