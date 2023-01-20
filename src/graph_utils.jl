begin
    using Graphs
    using SimpleWeightedGraphs
    using GraphIO
end

function read_edge_list_weighted(filename)
	csv = CSV.read(filename, DataFrame; header=false, delim=" ")
	labels = unique(sort(vcat(csv[:, 1], csv[:, 2])))
	n = length(labels)
	g = SimpleWeightedGraph(n)
	for row in eachrow(csv)
		if length(row) < 3
			@show "bad row"
			@show row
			continue
		end
		weight = row[3]
		if weight == 0
			weight = 0.000001
		end
		if findfirst(x -> x == row[2], labels) == nothing
			@show labels
			@show row
		end
		point_a = findfirst(x -> x == row[1], labels)
		point_b = findfirst(x -> x == row[2], labels)
		
		add_edge!(g, point_a, point_b, row[3])
		add_edge!(g, point_b, point_a, row[3])
	end

	g, labels
end

function getindex(elem, v)
    findfirst(x -> x == elem, v)
end

function subgraph(g, labels)
    g_res = SimpleWeightedGraph(length(labels))
    fa = zeros(nv(g))
    for point in labels
        fa[point] = 1
    end
    for edge in edges(g)
        if fa[src(edge)] == 1 && fa[dst(edge)] == 1
            add_edge!(g_res, getindex(src(edge), labels), getindex(dst(edge), labels), weight(edge))
        end
    end

    g_res
end