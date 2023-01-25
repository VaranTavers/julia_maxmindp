function combine_mmdp_algs(alg1, alg2)
	(n, k, min_dists) -> begin
		res = alg1(n, k, min_dists)
		alg2(n, k, min_dists, res)
	end
end