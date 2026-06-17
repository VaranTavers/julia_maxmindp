function mut_combiner(n, v, min_dists, fs, probs)
    chosen_fs = sample(probs)

    fs[chosen_fs](n, v, min_dists)
end