function single_pop(m, sums)
    res = zeros(Int64, m)
    sums_c = sums ./ sum(sums)

    for i in 1:m
        res[i] = sample(sums_c)
        sums_c[res[i]] = 0
        sums_c ./= sum(sums_c)
    end

    res
end

function high_sum(popSize, m, minDists)
    sums = sum(minDists; dims=2)[:]

    [single_pop(m, sums) for _ = 1:popSize]
end