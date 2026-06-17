# Swap some nodes with close by roulette
function mutation_change_to_close(n, v, min_dists, numberOfChanges=1)
    res = copy(v)

    for _ in 1:numberOfChanges
        toChange = rand(1:length(res))
        probs = (maximum(min_dists) .- min_dists[res[toChange], :])
        probs[res[toChange]] = 0
        probs ./= sum(probs)

        newNode = res[toChange]
        while newNode ∈ res
            newNode = sample(probs)
        end

        res[toChange] = newNode
    end


    res
end
