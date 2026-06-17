function g(Φ)
    1 / sqrt(1 + 3 * Φ^2 / π^2)
end

function E(μ, μ_j, Φ_j)
    1 / (1 + exp(-g(Φ_j) * (μ - μ_j)))
end

function f(x, Δ, Φ, v, a, τ)
    exp(x) * (Δ^2 - Φ^2 - v - exp(x)) / 2(Φ^2 + v + exp(x))^2 - (x - a) / τ^2
end

function calculate_new_sigma(Δ, Φ, v, τ, σ)
    ϵ = 0.000001
    a = log(σ^2)
    A = a
    B = 0

    local_f(x) = f(x, Δ, Φ, v, a, τ)
    if Δ^2 > Φ^2 + v
        B = log(Δ^2 - Φ^2 - v)
    else
        k = 1
        while local_f(a - k * τ) < 0
            k += 1
        end
        B = a - k * τ
    end

    fA = local_f(A)
    fB = local_f(B)
    while abs(B - A) > ϵ
        C = A + (A - B) * fA / (fB - fA)
        fC = local_f(C)
        if fB * fC <= 0
            A = B
            fA = fB
        else
            fA /= 2
        end
        B = C
        fB = fC
    end

    exp(A / 2)
end

# Based on: https://www.sciencedirect.com/science/article/pii/S002002551400276X
# and:      https://www.glicko.net/glicko/glicko2.pdf
function chess_ranking(dfs, num_tournaments, num_challenges, get_pairings)
    num_competitors = length(dfs)

    τ = 0.2 # Constraints the volatility
    ratings = 1500 .* ones(num_competitors)
    rating_deviations = 350 .* ones(num_competitors)
    rating_volatilities = 0.06 .* ones(num_competitors) # σ

    for i in 1:num_tournaments
        μ = (ratings .- 1500) ./ 173.7178
        Φ = rating_deviations ./ 173.7178

        pairings, s_s = get_pairings(dfs, num_competitors, num_challenges)



        #@show pairings, s_s
        v = [sum([g(Φ[j])^2 * E(μ[i], μ[j], Φ[j]) * (1 - E(μ[i], μ[j], Φ[j])) for j in pairings[i]])^-1 for i in 1:num_competitors]


        Δ = [v[i] * sum([g(Φ[j]) * (s_s[i][k] - E(μ[i], μ[j], Φ[j])) for (k, j) in enumerate(pairings[i])]) for i in 1:num_competitors]


        rating_volatilities = [calculate_new_sigma(Δ[i], Φ[i], v[i], τ, rating_volatilities[i]) for i in 1:num_competitors]
        Φ_pre = [sqrt(Φ_i^2 + σ^2) for (Φ_i, σ) in zip(Φ, rating_volatilities)]

        Φ = [1 / sqrt(1 / Φ_i^2 + 1 / v[i]) for (i, Φ_i) in enumerate(Φ_pre)]
        μ = μ + [Φ[i]^2 * sum([g(Φ[j]) * (s_s[i][k] - E(μ[i], μ[j], Φ[j])) for (k, j) in enumerate(pairings[i])]) for i in 1:num_competitors]

        ratings = 173.7178 .* μ .+ 1500
        rating_deviations = 173.7178 .* Φ
    end

    ratings, rating_deviations
end


# Runs the chess algorithm for each node pair in each graph, each competitor against all others
# The previous version had a weakness because the tournaments were conducted in order, later tournaments potentially had bigger impacts than previous ones
# For example if in the first 3 tournamens A won over B, but in the last one A lost, the result was markedly different than in the case of losing the first and winning the other 3
# To mitigate this problem we randomize the order of the tournaments.
function chess_ranking_all_single_time(dfs, get_all_pairings, tournaments_per_row)
    num_competitors = length(dfs)

    τ = 0.2 # Constraints the volatility, which is needed because it would get out of hand really fast
    ratings = 1500 .* ones(num_competitors)
    rating_deviations = 350 .* ones(num_competitors)
    rating_volatilities = 0.06 .* ones(num_competitors) # σ

    tournaments = []

    for row in 1:possible_rows
        append!(tournaments, [row for _ in 1:tournaments_per_row])
    end

    shuffle!(tournaments)
    for row in tournaments
        μ = (ratings .- 1500) ./ 173.7178
        Φ = rating_deviations ./ 173.7178


        pairings, s_s = get_all_pairings(dfs, row)


        v = [sum([g(Φ[j])^2 * E(μ[i], μ[j], Φ[j]) * (1 - E(μ[i], μ[j], Φ[j])) for j in pairings[i]])^-1 for i in 1:num_competitors]

        Δ = [v[i] * sum([g(Φ[j]) * (s_s[i][k] - E(μ[i], μ[j], Φ[j])) for (k, j) in enumerate(pairings[i])]) for i in 1:num_competitors]

        rating_volatilities = [calculate_new_sigma(Δ[i], Φ[i], v[i], τ, rating_volatilities[i]) for i in 1:num_competitors]

        Φ_pre = [sqrt(Φ_i^2 + σ^2) for (Φ_i, σ) in zip(Φ, rating_volatilities)]


        Φ = [1 / sqrt(1 / Φ_i^2 + 1 / v[i]) for (i, Φ_i) in enumerate(Φ_pre)]
        μ = μ + [Φ[i]^2 * sum([g(Φ[j]) * (s_s[i][k] - E(μ[i], μ[j], Φ[j])) for (k, j) in enumerate(pairings[i])]) for i in 1:num_competitors]


        ratings = 173.7178 .* μ .+ 1500
        rating_deviations = 173.7178 .* Φ


    end

    ratings, rating_deviations
end


# Runs the chess algorithm for each node pair in each graph, each competitor against all others
function chess_ranking_single_motif_single_time(dfs_orig, get_all_pairings, motif, tournaments_per_row)
    dfs = map(x -> Dict(motif => x[motif]), dfs_orig)
    chess_ranking_all_single_time(dfs, get_all_pairings, tournaments_per_row)
end


# Runs the chess algorithm for each node pair in each graph, each competitor against all others
function chess_ranking_single_motif_random(dfs_orig, get_pairings, motif, num_tournaments, num_challenges)
    dfs = map(x -> Dict(motif => x[motif]), dfs_orig)
    chess_ranking(dfs, num_tournaments, num_challenges, get_pairings)
end