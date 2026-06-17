function random_init(popSize, m, _minDists)
    [randperm(nv(g))[1:m] for _ = 1:popSize]
end