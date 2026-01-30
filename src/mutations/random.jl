function has_duplicates(v, newPointId)
  for i in eachindex(v)
    if i != newPointId && v[i] == v[newPointId]
      return true
    end
  end
  false
end

function mutate(n, v)
  changeId = rand(1:length(v))
  v[changeId] = rand(1:n)
  while has_duplicates(v, changeId)
    v[changeId] = rand(1:n)
  end

  v
end