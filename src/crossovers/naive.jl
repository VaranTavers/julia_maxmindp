function crossoverNaive(v1, v2)
  v3 = copy(v1)
  append!(v3, v2)
  v3 = unique(sort(v3))
  collect(shuffle(v3))[1:length(v1)]
end