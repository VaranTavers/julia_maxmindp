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

function maxmindp_genetic_dist(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate; trace = false)
	people = [maxmindp_random(n, k) for i in 1:populationSize]
	maxmindp_genetic_dist3(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate, people, trace=trace)
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

sample(weights) = findfirst(cumsum(weights) .> rand())

function maxmindp_genetic_dist3(n, min_dists, k, numberOfIterations, populationSize, mutationRate, crossoverRate, people; trace = false)
	max_val = calculate_mindist(people[1], min_dists)
	max_vec = copy(people[1])
	start = copy(people[1])
	probs = ones(length(people)) ./ length(people)
	for i in 1:numberOfIterations
		for j in length(people):populationSize
		#	 push!(people, crossover(people[rand(1:length(people))],  people[rand(1:length(people))]))
			
			push!(people, crossover(people[sample(probs)],  people[sample(probs)]))
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
		probs = copy(scores[score_sorted])[1:halfOfPeople]
		s = sum(probs)
		probs = collect(map(x->x/s, probs))
		#@show probs
		#@show cumsum(probs)
		push!(people, start)

		if trace
			@show max_val, i, sort(max_vec)
		end
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

function mutationFromSBTSRand(n, v, min_dists)
	r = 12
	candidates = get_candidates(n, v)
	scores_new = collect(map(x -> calculate_sumdp(x, v, min_dists), candidates))
	scores_old = collect(map(x -> calculate_sumdp(x, v, min_dists), v))
	new_sorted = sortperm(scores_new, rev=true)
	old_sorted = sortperm(scores_old)
	v[rand(old_sorted[1:r])] = candidates[rand(new_sorted[1:r])]

	v
end

function mutationFromSBTSRoulette(n, v, min_dists)
	r = 12
	candidates = get_candidates(n, v)
	scores_new = collect(map(x -> calculate_sumdp(x, v, min_dists), candidates))
	scores_old = collect(map(x -> calculate_sumdp(x, v, min_dists), v))
	new_sorted = sortperm(scores_new, rev=true)
	old_sorted = sortperm(scores_old)
	s_new_sorted = copy(new_sorted[1:r]) ./ sum(new_sorted[1:r])
	s_old_sorted = copy(new_sorted[1:r]) ./ sum(new_sorted[1:r])
	v[sample(s_old_sorted[1:r])] = candidates[sample(s_new_sorted[1:r])]

	v
end

struct GeneticSettings
	populationSize
	mutationRate
	crossoverRate
	elitRate
end

struct RunSettings
	minDists
	k
	numberOfIterations
end

function crossoverRoulette(chromosomes, fitness)
	rouletteWheel = fitness ./ sum(fitness)

	crossover(
		chromosomes[sample(rouletteWheel)],
		chromosomes[sample(rouletteWheel)]
	)
end

'''

'''
function maxmindp_genetic_dist4(runS::RunSettings, gaS::GeneticSettings, chromosomes; trace = false)
	# Initializing values and functions for later use
	n, _ = size(runS.minDists)
	calcFitness(x) = calculate_mindist(x, runS.minDists)
	runMutation(x) = rand() < gaS.mutationRate ? mutationFromSBTS(n, x, runS.minDists) : x
	chromosomes = deepcopy(chromosomes)

	# Initializing global maximum as one of the given chromosome
	maxVal = calculate_mindist(chromosomes[1], runS.minDists)
	maxVec = copy(chromosomes[1])

	fitness = collect(
		map(calcFitness, chromosomes)
	)
	for i in 1:runS.numberOfIterations
		# Creating p_c% new individuals with the crossover
		# operator, choosing parents based on fitness.
		newChromosomes = [
			crossoverRoulette(chromosomes, fitness)
			for _ in 1:Int(ceil(n * gaS.crossoverRate))
		]
		newFitness = collect(
			map(calcFitness, chromosomes)
		)
		
		# Add them to the chromosome pool
		append!(chromosomes, newChromosomes)
		append!(fitness, newFitness)

		# Mutating individuals
		chromosomes = collect(
			map(runMutation, chromosomes)
		)

		# Recalculating fitness for new individuals
		fitness = collect(
			map(calcFitness, chromosomes)
		)

		# Sorting fitness scores
		fitnessSorted = sortperm(fitness, rev=true)

		# Choosing the elit
		elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
		elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
		elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

		# Choosing the rest randomly from the others
		restNumber = gaS.populationSize - elitNumber
		restIds = [
			rand(fitnessSorted[elitNumber+1:end])
			for _ in 1:restNumber
		]
		restChromosomes = map(x -> copy(chromosomes[x]), restIds)
		restFitness = map(x -> fitness[x], restIds)

		chromosomes = vcat(elitChromosomes, restChromosomes)
		fitness = vcat(elitFitness, restFitness)

		maxVec = copy(chromosomes[1])
		maxVal = fitness[1]

		if trace
			@show maxVal, i, sort(maxVec)
		end
	end

	maxVec
end
