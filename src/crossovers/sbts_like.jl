function calculateSumdp(newPoint, v, minDists)
  sum(map(x -> minDists[x, newPoint], v))
end

function SBTSLikeCrossoverOne(left, right, minDists)

  scoresLeft = collect(map(x -> calculate_sumdp(x, left, minDists), left))
  scoresRight = collect(map(x -> calculate_sumdp(x, right, minDists), right))
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

function crossoverSBTSLike(chromosomes, fitness, minDists)
  rouletteWheel = fitness ./ sum(fitness)

  SBTSLikeCrossoverOne(
    chromosomes[sample(rouletteWheel)],
    chromosomes[sample(rouletteWheel)],
    minDists
  )
end