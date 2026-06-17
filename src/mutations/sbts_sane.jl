# Swap the worst performing from IN with the best from OUT
function mutationSBTSSane(n, v, min_dists, numberOfPossibilitiesPerChange=5, numberOfChanges=1, partFitF=DispersionProblems.calculateSumdp)
  remaining = shuffle(v)[numberOfChanges+1:end]

  candidates = shuffle(get_candidates(n, remaining))[1:min(numberOfPossibilitiesPerChange * numberOfChanges, n - length(remaining))]

  scores_new = collect(map(x -> partFitF(x, remaining, min_dists), candidates))

  scores_index_sorted = sortperm(scores_new, rev=true)

  vcat(remaining, candidates[scores_index_sorted][1:numberOfChanges])
end
