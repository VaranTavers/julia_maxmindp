using CSV
using DataFrames
using Graphs
using GraphIO
using WeightedEdgeListFormat
using SimpleWeightedGraphs
using HypothesisTests
using Plots
using Statistics
using StatsPlots

gr()

# Extracts maxima from results_compiled files

function compiled_file_name(files)
    return "results_compiled.csv"
end

function calculate_mindist(vertices, min_distances)
    dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)

    minimum(map(sum, dist_sums))
end


graphs_folder_files = readdir("../mmdp_graphs/")

gs = Dict()

for g in graphs_folder_files
    if g[end-3:end] == ".dat"

        gs[g] = loadgraph("../mmdp_graphs/$(g)", WELFormat(" "))
    end
end

folders = readdir("../results/")

configs = Dict()

for folder in folders
    if isdir("../results/$(folder)")
        inner_dict = Dict()
        files = readdir("../results/$(folder)")

        for file in files
            if isfile("../results/$(folder)/$(file)") && file != compiled_file_name(files)
                df = CSV.read("../results/$(folder)/$(file)", DataFrame; header = true)
                graph_name = file[12:end-4]
                if graph_name in collect(keys(gs))
                    g = gs[graph_name]
                    inner_dict[graph_name] = collect(
                        map(
                            x -> round.(calculate_mindist(x, g.weights); digits = 6),
                            eachcol(df),
                        ),
                    )
                end
            end
        end

        configs[folder] = inner_dict
    end
end


point_sum = zeros(length(keys(configs)), length(keys(configs)))

configs_keys = sort(collect(keys(configs)))

#=
for k1 in configs_keys
  k1_keys = collect(keys(configs[k1]))
  if !("02Type2.2_n500m50.dat" in k1_keys)
    @show k1
    @show configs[k1]
  end
end
=#

for g in keys(gs)

    conf_mat = [
        pvalue(MannWhitneyUTest(configs[k1][g], configs[k2][g])) < 0.05 ?
        (mean(configs[k1][g]) > mean(configs[k2][g]) ? 1 : -1) : 0 for
        k1 in configs_keys, k2 in configs_keys
    ]
    conf_mat_d = [
        pvalue(MannWhitneyUTest(configs[k1][g], configs[k2][g])) < 0.05 ? 1 : 0 for
        k1 in configs_keys, k2 in configs_keys
    ]
    df = DataFrame(x = Int[], y = Float64[])

    for (i, k1) in enumerate(configs_keys)
        for v in configs[k1][g]
            push!(df, (i, v))
        end
    end

    global point_sum += conf_mat
    savefig(
        Plots.heatmap(
            conf_mat_d,
            yflip = true,
            aspect_ratio = 0.5,
            color = [:white, :black],
            showaxis = :x,
            xticks = 1:length(configs_keys),
            title = split(g, "_n")[1],
        ),
        "../images/heat_$(g).pdf",
    )
    savefig(
        Plots.heatmap(
            conf_mat,
            yflip = true,
            aspect_ratio = 0.5,
            showaxis = :x,
            xticks = 1:length(configs_keys),
            title = split(g, "_n")[1],
        ),
        "../images/heat2_$(g).pdf",
    )

    #=if occursin("APOM_02", g)
        df[:, :y] .= round.(df[:, :y]; digits = 6)
        @show df[:, :y]

    end=#

    savefig(
        boxplot(
            df[:, :x],
            df[:, :y],
            line = (2, :black),
            fill = (0.3, :orange),
            legend = false,
            xticks = 1:length(configs_keys),
            ylabel = "fitness value",
            title = split(g, "_n")[1],
        ),
        "../images/box_$(g).pdf",
    )



    savefig(
        scatter(
            df[:, :x],
            df[:, :y],
            legend = false,
            xticks = 1:length(configs_keys),
            title = split(g, "_n")[1],
        ),
        "../images/scatter_$(g).pdf",
    )
    df = DataFrame(conf_mat, configs_keys)
    df[!, "Config"] = configs_keys
    CSV.write("../images/$(g).csv", df)


end

point_sum
savefig(
    heatmap(
        point_sum,
        rev = true,
        aspect_ratio = 0.5,
        showaxis = :x,
        xticks = 1:length(configs_keys),
    ),
    "../images/sum.pdf",
)
sum_df = DataFrame(point_sum, configs_keys)
sum_df[!, "Config"] = configs_keys

sum_of_rows = sum(point_sum, dims = 2)
best_version = argmax(sum_of_rows)[1]
best_value = maximum(sum_of_rows)
@show sum_of_rows
sum_df[!, "Sum"] = sum_of_rows[:]

CSV.write("../images/sum.csv", sum_df)


@show collect(enumerate(configs_keys))
# @show sum_of_rows
# @show point_sum
@show best_version, best_value, configs_keys[best_version]
