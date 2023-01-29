begin
	using Graphs
	using SimpleWeightedGraphs
	using Evolutionary
	using Base.Iterators
	using Statistics
	using Random
end


include("graph_utils.jl")

function f(x, k, min_dists)
	- calculate_mindist(x[1:k], min_dists)
end

function rest_of_things(x, n)
	a = collect(1:n)
	d = setdiff(Set(a), Set(x))
	x2 = copy(x)
	append!(x2, d)

	x2
end

function mmdp_evolutionary(n, k, min_dists)
	Evolutionary.optimize(
		x -> f(x, k, min_dists), randperm(n),
		GA(populationSize = 100, selection = roulette,
			 crossover = OX2, mutation = swap2, metrics=[]), Evolutionary.Options(iterations=100, show_trace=true))
end

function mmdp_evolutionary2(n, k, min_dists, start)
	Evolutionary.optimize(
		x -> f(x, k, min_dists), rest_of_things(start, n),
		GA(populationSize = 100, selection = roulette,
			 crossover = OX2, mutation = swap2, metrics=[]), Evolutionary.Options(iterations=100, show_trace=true))
end