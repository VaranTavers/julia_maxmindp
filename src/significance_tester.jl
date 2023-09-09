using CSV
using DataFrames
using Graphs
using GraphIO
using WeightedEdgeListFormat
using SimpleWeightedGraphs
using HypothesisTests
using Plots
using Statistics

# Extracts maxima from results_compiled files

function compiled_file_name(files)
  return "results_compiled.csv"
end

function calculate_mindist(vertices, min_distances)
  dist_sums = map(i -> map(j -> min_distances[i, j], vertices), vertices)

  minimum(map(sum, dist_sums))
end


graphs_folder_files = readdir("mmdp_graphs/")

gs = Dict()

for g in graphs_folder_files
  if g[end-3:end] == ".dat"

    gs[g] = loadgraph("mmdp_graphs/$(g)", WELFormat(" "))
  end
end

folders = readdir("results/")

configs = Dict()

for folder in folders
  if isdir("results/$(folder)")
    inner_dict = Dict()
    files = readdir("results/$(folder)")

    for file in files
      if isfile("results/$(folder)/$(file)") && file != compiled_file_name(files)
        df = CSV.read("results/$(folder)/$(file)", DataFrame; header=true)
        graph_name = file[12:end-4]
        if graph_name in collect(keys(gs))
          g = gs[graph_name]
          inner_dict[graph_name] = collect(map(x -> calculate_mindist(x, g.weights), eachcol(df)))
        end
      end
    end

    configs[folder] = inner_dict
  end
end


point_sum = zeros(length(keys(configs)), length(keys(configs)))

configs_keys = collect(keys(configs))

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
    pvalue(MannWhitneyUTest(
      configs[k1][g],
      configs[k2][g]
    )) < 0.05 ? (mean(configs[k1][g]) > mean(configs[k2][g]) ? 1 : -1) : 0
    for k1 in configs_keys,
    k2 in configs_keys
  ]
  global point_sum += conf_mat
  png(heatmap(conf_mat, rev=true, aspect_ratio=1), "images/$(g).png")

  df = DataFrame(conf_mat, configs_keys)
  df[!, "Config"] = configs_keys
  CSV.write("images/$(g).csv", df)


end

point_sum
png(heatmap(point_sum, rev=true, aspect_ratio=1), "images/sum.png")
sum_df = DataFrame(point_sum, configs_keys)
sum_df[!, "Config"] = configs_keys

sum_of_rows = sum(point_sum, dims=2)
best_version = argmax(sum_of_rows)[1]
best_value = maximum(sum_of_rows)
@show sum_of_rows
sum_df[!, "Sum"] = sum_of_rows[:]

CSV.write("images/sum.csv", sum_df)


@show collect(enumerate(configs_keys))
# @show sum_of_rows
# @show point_sum
@show best_version, best_value, configs_keys[best_version]
