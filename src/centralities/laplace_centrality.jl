begin
    using Graphs
    using SimpleWeightedGraphs
end

# Implemented following this article: https://andrewpwheeler.com/2015/07/29/laplacian-centrality-in-networkx-python/

function lap_energy(g)
	w = weights(g)
	degrees = sum(w, dims=1)

	sum(map(x->x^2, degrees)), 2*sum(map(e -> e.weight^2, edges(g)))
end

function cw(g, node)
	cw_res = 0
	sub = 0
	w = weights(g)

	for i in 1:nv(g)
		w_edge = w[node, i]
		if w_edge != 0
			od = sum(w[i,:])
			sub += -(od.^2) + (od .- w_edge).^2
			cw_res += w_edge^2
		end
	end

	cw_res, sub
end

function lap_inner(g, i, w)
	d2 = sum(w[i,:])
	cw_res, sub = cw(g, i)
	fin = d2^2 - sub + 2 * cw_res
end

function lap_cent_weighted(g; norm=false)
	lap_cent_weighted(g, 1:nv(g); norm)
end

function lap_cent_weighted(g, nbunch; norm=false)
	w = weights(g)
	den = 1
	if norm
		d1, dw1 = lap_energy(g)
		den = d1+dw1
	end
	collect(map(x->lap_inner(g, x, w) / den, nbunch))
end