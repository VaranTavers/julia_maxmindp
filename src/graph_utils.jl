begin
    using Graphs
    using SimpleWeightedGraphs
    using GraphIO
		using CSV
		using DataFrames
end

struct WELFormat <: Graphs.AbstractGraphFormat
	delim
	WELFormat() = new(",")
	WELFormat(delim) = new(delim)
end

function Graphs.loadgraph(io::IO, gname::String, f::WELFormat)
	g, _ = read_edge_list_weighted(io; delim=f.delim)

	g
end

Graphs.loadgraphs(io::IO, f::WELFormat) = loadgraph(io, "...", f)

function read_edge_list_weighted(filename; delim=" ")
	csv = CSV.read(filename, DataFrame; header=false, delim=delim)
	labels = unique(sort(vcat(csv[:, 1], csv[:, 2])))
	n = length(labels)
	g = SimpleWeightedGraph(csv[:, 1], csv[:, 2], csv[:, 3]; combine=max)

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

function calculate_mindist(vertices, min_distances)
	
	dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)

	minimum(map(sum,dist_sums))
end
