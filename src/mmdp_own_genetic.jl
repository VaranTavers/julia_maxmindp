begin
	using Graphs
	using SimpleWeightedGraphs
	using Folds
	using Base.Iterators
	using Random
end

include("graph_utils.jl")

function maxmindp_random(n, k)
	randperm(n)[1:k]
end

function has_duplicates(v, newPointId)
	for i in 1:length(v)
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

function crossover(v1, v2)
	v3 = copy(v1)
	append!(v3, v2)
	v3 = unique(sort(v3))
	collect(shuffle(v3))[1:length(v1)]
end

function maxmindp_genetic_dist(n, min_dists, k, numberOfIterations, numberOfPeople, mutationRate, crossoverRate)
	people = [maxmindp_random(n, k) for i in 1:numberOfPeople]
	maxmindp_genetic_dist2(n, min_dists, k, numberOfIterations, numberOfPeople, mutationRate, crossoverRate, people)
end

function maxmindp_genetic_dist2(n, min_dists, k, numberOfIterations, numberOfPeople, mutationRate, crossoverRate, people)
	max_val = calculate_mindist(people[1], min_dists)
	max_vec = people[1]
	for i in 1:numberOfIterations
		people = collect(map(x -> rand() < mutationRate ? mutate(n, x) : x, people))
		
		scores = map(x -> calculate_mindist(x, min_dists), people)
		score_sorted = sortperm(scores)

		if scores[score_sorted[1]] > max_val
			max_val = scores[score_sorted[1]]
			max_vec = copy(people[score_sorted[1]])
		end

		halfOfPeople = Int32(floor(numberOfPeople/2))
		people = collect(map(x->people[x], score_sorted[end-halfOfPeople+1:end]))

		for j in halfOfPeople+1:numberOfPeople
			push!(people, crossover(people[rand(1:halfOfPeople)],  people[rand(1:halfOfPeople)]))
		end
	end

	max_vec, max_val
end