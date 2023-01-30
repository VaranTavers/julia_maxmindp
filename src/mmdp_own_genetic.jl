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

function maxmindp_genetic_dist(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate)
	people = [maxmindp_random(n, k) for i in 1:populationSize]
	maxmindp_genetic_dist3(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate, people)
end

function maxmindp_genetic_dist2(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate, people)
	max_val = calculate_mindist(people[1], min_dists)
	max_vec = people[1]
	for i in 1:numberOfIterations
		people = collect(map(x -> rand() < mutationRate ? mutate(n, x) : x, people))
		
		scores = map(x -> calculate_mindist(x, min_dists), people)
		score_sorted = sortperm(scores, rev=true)

		if scores[score_sorted[1]] > max_val
			max_val = scores[score_sorted[1]]
			max_vec = copy(people[score_sorted[1]])
		end

		halfOfPeople = Int32(floor(populationSize/2))
		people = collect(map(x->people[x], score_sorted[1:halfOfPeople]))

		for j in halfOfPeople+1:populationSize
			push!(people, crossover(people[rand(1:halfOfPeople)],  people[rand(1:halfOfPeople)]))
		end
	end

	max_vec
end

function maxmindp_genetic_dist3(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate, people)
	max_val = calculate_mindist(people[1], min_dists)
	max_vec = copy(people[1])
	start = copy(people[1])
	for i in 1:numberOfIterations
		for j in length(people):populationSize
			push!(people, crossover(people[rand(1:length(people))],  people[rand(1:length(people))]))
		end
		# @show people
		scores = collect(map(x -> calculate_mindist(x, min_dists), people))
		#@show scores
		score_sorted = sortperm(scores, rev=true)

		if scores[score_sorted[1]] > max_val
			max_val = copy(scores[score_sorted[1]])
			max_vec = copy(people[score_sorted[1]])
		end

		# @show max_val, calculate_mindist(max_vec, min_dists), max_vec

		people = collect(map(x -> rand() < mutationRate ? mutationFromSBTS(n, x, min_dists) : x, people))

		halfOfPeople = Int32(floor(populationSize*(1 - crossoverRate)))
		people = collect(map(x->people[x], score_sorted[1:halfOfPeople]))
		push!(people, start)
	end

	max_vec
end

function calculate_sumdp(new_point, v, min_dists)
	sum(map(x -> min_dists[x, new_point], v))
end

function get_candidates(n, v)
	a = collect(1:n)
	
	collect(setdiff(Set(a), Set(v)))
end

function mutationFromSBTS(n, v, min_dists)
	candidates = get_candidates(n, v)
	scores_new = collect(map(x -> calculate_sumdp(x, v, min_dists), candidates))
	scores_old = collect(map(x -> calculate_sumdp(x, v, min_dists), v))

	v[argmin(scores_old)] = candidates[argmax(scores_new)]

	v
end