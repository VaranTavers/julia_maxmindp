module ACO

begin
    using Graphs
    using DataFrames
    using CSV
    using SimpleWeightedGraphs
    using Folds
    using ConcurrentCollections

end


struct ACOSettings
    α::Real
    β::Real
    number_of_ants::Integer
    ρ::Real
    ϵ::Real
    starting_pheromone_ammount::Real
    ACOSettings() = new(1, 1, 50, 0.7, 0.3, 1)
    ACOSettings(α, β, n_a, ρ, ϵ, start_ph) = new(α, β, n_a, ρ, ϵ, start_ph)
end

struct RunSettings
    k::Int
    fitF::Function
    partFitF::Function
    max_number_of_evaluations::Integer
    logging_file
end

# Get chosen point
function get_chosen_point_row(row, i, r)
    if maximum(row) == 0
        @show row
        return i
    end

    findfirst(row .> r)
end

function get_chosen_point(pM, i, r)
    get_chosen_point_row(pM[i, :], i, r)
end



# Constructs a new solution in the form of a list of nodes visited: [1, 2, 3] 
function generate_s(k, pM, pS)
    s = zeros(Int32, k)
    n, _ = size(pM)
    visited = [false for _ ∈ 1:n]


    s[1] = get_chosen_point_row(pS, 1, rand())
    visited[s[1]] = true
    i = 1
    while i < k
        j = 0
        res = s[1]
        while visited[res] && j < 100
            res = get_chosen_point(pM, s[i], rand())
            j += 1
        end
        # Stuck between a rock and a hard place
        if j == 100
            # @show "stuck"
            # @show visited
            # @show s
            return nothing
        end

        # Append to solution
        s[i+1] = res
        visited[res] = true
        i += 1
    end

    s
end



function ACO_test(vars::ACOSettings, runS::RunSettings, η, τ, τ_start, adj_mat; use_folds=true)
    #Set parameters and initialize pheromone traits.
    evaluations = 0

    sgb = [i for i ∈ 1:runS.k]
    sgb_val = -1000
    τ_max = vars.starting_pheromone_ammount
    τ_min = 0
    fitnessCache = ConcurrentDict{Vector{Integer},Float64}()


    #@show runS.start_p, runS.end_p
    # While termination condition not met
    while evaluations <= runS.max_number_of_evaluations
        #@show i
        #@time 
        begin
            # Precomputing this
            η_d_sq = η .^ vars.β


            # Construct new solution S
            # Precomputing the probabilities results in a 2s time improvement.
            probM = τ .^ vars.α .* η_d_sq
            #probM = [τ[i, j] ^ vars.α * η[i, j] ^ vars.β for i in 1:n, j in 1:n]
            probM ./= sum(probM, dims=2)
            probM = cumsum(probM, dims=2)

            probS = τ_start
            probS ./= sum(probS)
            probS = cumsum(probS)

            S = []
            if use_folds
                S = Folds.map(
                    x -> generate_s(runS.k, probM, probS),
                    1:vars.number_of_ants,
                )
            else
                S = collect(map(
                    x -> generate_s(runS.k, probM, probS),
                    1:vars.number_of_ants,
                ))
            end


            evaluations += count(S .!== nothing)


            #@show length(unique(sort(sort.(S)))) / length(S)
            fitness = collect(map(x -> runS.fitF(x, adj_mat), S)) # ; fitnessCache

            # Update iteration best
            maxIndex = argmax(fitness)
            (sib, sib_val) = (deepcopy(S[maxIndex]), fitness[maxIndex])
            if sib_val > sgb_val && sib !== nothing
                sgb_val = copy(sib_val)
                sgb = copy(sib)

                # Compute pheromone trail limits
                τ_max = (sgb_val + 1) / (1 - vars.ρ)
                τ_min = vars.ϵ * τ_max

            elseif sib === nothing
                @show sib_val, sgb_val
            end
            if !isempty(runS.logging_file)
                #@show logging_file
                logdf = DataFrames.DataFrame(sib=[sib === nothing ? [] : sib], sib_val=[sib_val === nothing ? -1 : sib_val], sgb=[sgb === nothing ? [] : sgb], sgb_val=[sgb_val === nothing ? -1 : sgb_val])
                CSV.write(runS.logging_file, logdf; append=true)
            end
            # Update pheromone trails
            # TODO: test with matrix sum
            τ .*= (1 - vars.ρ)
            τ_start .*= (1 - vars.ρ)

            if sib !== nothing
                τ_start[sib[1]] += sib_val
            end

            for (sol, fit) in zip(S, fitness)
                for (s, e) in zip(sib, filter(x -> x > 0, sol[2:end]))
                    τ[s, e] += fit
                    τ[e, s] += fit
                end
            end

            τ = min.(τ, τ_max)
            τ = max.(τ, τ_min)

            τ_start = min.(τ_start, τ_max)
            τ_start = max.(τ_start, τ_min)

        end
    end

    #@show τ
    sgb, sgb_val
end

function ACO_preprocessing(vars::ACOSettings, runS::RunSettings, adj_mat; use_folds=true)
    n, m = size(adj_mat)
    τ = ones((n, m)) * vars.starting_pheromone_ammount
    τ_start = ones(n) * vars.starting_pheromone_ammount
    η = deepcopy(adj_mat)

    η ./= sum(η)

    ACO_test(vars, runS, η, τ, τ_start, adj_mat; use_folds=use_folds)
end

end
