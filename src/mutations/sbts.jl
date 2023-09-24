
function calculate_sumdp(new_point, v, min_dists)
  sum(map(x -> min_dists[x, new_point], v))
end

function get_candidates(n, v)
  a = collect(1:n)

  collect(setdiff(Set(a), Set(v)))
end

sample(weights) = findfirst(cumsum(weights) .> rand())

# IN - means the group of nodes in the solution
# OUT - means the group of nodes NOT in the solution

# Chooses the worst from IN
function sumdpGreedyIN(scores)
  argmin(scores)
end

# Chooses by Roulette wheel from bottom p% IN
function sumdpRouletteIN(scores; p=1)
  r = Int32(ceil(length(scores) * p))

  sorted = sortperm(scores)
  chosen_scores = scores[sorted][1:r]
  chosen_probs = sum(chosen_scores) ./ chosen_scores
  chosen_probs = chosen_probs ./ sum(chosen_probs)

  sorted[sample(chosen_probs)]
end

# Chooses randomly from bottom p% IN
function sumdpRandomIN(scores; p=1)
  r = Int32(ceil(length(scores) * p))
  sorted = sortperm(scores)

  sorted[rand(1:r)]
end

# OUT

# Chooses the best from OUT
function sumdpGreedyOUT(scores)
  argmax(scores)
end

# Chooses by Roulette wheel from top p% of OUT
function sumdpRouletteOUT(scores; p=1)
  r = Int32(ceil(length(scores) * p))

  sorted = sortperm(scores, rev=true)
  chosen_scores = scores[sorted][1:r]
  chosen_probs = chosen_scores ./ sum(chosen_scores)

  sorted[sample(chosen_probs)]
end

# Chooses randomly from top p% of OUT
function sumdpRandomOUT(scores; p=1)
  r = Int32(ceil(length(scores) * p))
  sorted = sortperm(scores, rev=true)

  sorted[rand(1:r)]
end


# Swap the worst performing from IN with the best from OUT
function mutationSBTS(n, v, min_dists; in_f=sumdpGreedyIN, out_f=sumdpGreedyOUT)
  candidates = get_candidates(n, v)
  scores_new = collect(map(x -> calculate_sumdp(x, v, min_dists), candidates))
  scores_old = collect(map(x -> calculate_sumdp(x, v, min_dists), v))

  v[in_f(scores_old)] = candidates[out_f(scores_new)]

  v
end

function customMutationAlg(n, v, minDists, numOfMoves)
  scores_old = collect(map(x -> calculate_sumdp(x, v, minDists), v))
  chosen_probs = sum(scores_old) ./ scores_old
  chosen_probs = chosen_probs ./ sum(chosen_probs)
  for _ in 1:numOfMoves
    good = false
    i = 0
    j = 0
    while !good
      i = sample(chosen_probs)
      j = rand(1:n)
      good = minDists[v[i], j] > 0 && findfirst(x -> x == j, v) === nothing
    end
    v[i] = j
  end

  v
end