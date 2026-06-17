using Pkg

Pkg.activate(".")

using SafestRoutes
using SafestRoutes.SingleMultiobjective.Pareto
using CSV
using DataFrames
using Plots
using HypothesisTests
using Statistics
using StatsPlots
using Random


### MAIN PROGRAM

include("config/config_comp.jl")


dfs = []
# Load variations
variations = collect(filter(x -> (isdir("$(variations_folder_name)/$(x)") && x != "comparison"), readdir(variations_folder_name)))
variation_nums = [parse(Int, split(x, "_")[1]) for x in variations]
variations = variations[sortperm(variation_nums)]
@show variations


function compare_means(df_1, df_2)
    count(df_1[:, :mean] .> df_2[:, :mean])
end

function compare_maxs(df_1, df_2)
    count(df_1[:, :max] .> df_2[:, :max])
end



function compare_random(df_1, df_2, row)
    _r1, c1 = size(df_1)
    _r2, c2 = size(df_2)
    o1 = rand(2:(c1-4))
    o2 = rand(2:(c2-4))

    #@show row, df_1[row, 1], df_2[row, 1], df_1[row, o1], df_2[row, o2]
    if df_1[row, o1] == df_2[row, o2]
        return 0.5
    end
    if df_1[row, o1] > df_2[row, o2]
        return 1
    end

    0
end


function get_pairings(dfs, num_challengers, num_challenges)
    @assert num_challenges <= num_challengers - 1
    competitors = [[] for _ in 1:num_challengers]
    results = [[] for _ in 1:num_challengers]

    for i in 1:num_challengers
        while length(competitors[i]) < num_challenges
            new = rand(1:num_challengers)
            while new == i || new in competitors[i]
                new = rand(1:num_challengers)
            end

            #@show i, new
            possible_rows, _ = size(dfs[1])
            result = compare_random(dfs[i], dfs[new], rand(1:possible_rows))
            push!(competitors[i], new)
            push!(competitors[new], i)
            push!(results[i], result)
            push!(results[new], 1 - result)
        end
    end

    competitors, results
end

function get_all_pairings(dfs, row)
    num_challengers = length(dfs)
    competitors = [[] for _ in 1:num_challengers]
    results = [[] for _ in 1:num_challengers]

    for i in 1:num_challengers
        for new in (i+1):num_challengers
            result = compare_random(dfs[i], dfs[new], row)
            push!(competitors[i], new)
            push!(competitors[new], i)
            push!(results[i], result)
            push!(results[new], 1 - result)
        end
    end

    competitors, results
end



# General funtions

function get_comparison_df(dfs, comp_f)
    l = length(dfs)
    mat = zeros(l, l)

    for i in 1:l
        for j in 1:l
            if i != j
                mat[i, j] = comp_f(dfs[i], dfs[j])
            end
        end
    end

    comp_df = DataFrame()
    comp_df[!, :nodes] = collect(1:l)

    for (i, val) in enumerate(eachcol(mat))
        comp_df[!, :x] = val
        rename!(comp_df, :x => "$(i)")
    end

    comp_df
end

### MAIN PROGRAM

include("utils/chess_comp.jl")
include("utils/tikz_comparator_plots.jl")


if !isdir("$(variations_folder_name)/comparison")
    mkdir("$(variations_folder_name)/comparison")
end


comp_dfs = []

for variation in variations

    dfs_var = CSV.read("$(variations_folder_name)/$(result_file_name)", DataFrame)

    push!(comp_dfs, dfs_var)
end

comparison_folder_name = "$(variations_folder_name)/comparison/"
if !isdir(comparison_folder_name)
    mkdir(comparison_folder_name)
end

# Simple mean comparison
if run_mean_comp
    comp_df = get_comparison_df(comp_dfs, compare_means)
    CSV.write("$(comparison_folder_name)/mean_comp.csv", comp_df)
end

# Simple max comparison

if run_max_comp
    comp_df = get_comparison_df(comp_dfs, compare_maxs)
    CSV.write("$(comparison_folder_name)/max_comp.csv", comp_df)
end

if run_chess_random

    chess_result, confidence = chess_ranking(comp_dfs, 10000, length(comp_dfs) - 1, get_pairings)

    @show chess_result, confidence

    sorted_indices = sortperm(chess_result, rev=true)
    sorted_result = chess_result[sorted_indices]
    sorted_confidence = confidence[sorted_indices]

    chess_p = scatter(sorted_result, [10 * i for i in length(chess_result):-1:0], yticks=([10 * i for i in length(chess_result):-1:0], sorted_indices), xerror=sorted_confidence, title="Chess ranking (random tournament)", label=false)

    savefig(chess_p, "$(comparison_folder_name)/chess_random_res.pdf")
    open("$(comparison_folder_name)/chess_random_res.tex", "w") do io
        println(io, tikz_chess((chess_result, confidence), collect(1:length(chess_result)), title="Chess ranking (random tournament)"))
    end

end

if run_chess_all_single

    chess_result, confidence = chess_ranking_all_single_time(comp_dfs, get_all_pairings, tournaments_per_single)

    @show chess_result, confidence

    sorted_indices = sortperm(chess_result, rev=true)
    sorted_result = chess_result[sorted_indices]
    sorted_confidence = confidence[sorted_indices]

    chess_p = scatter(sorted_result, [10 * i for i in length(chess_result):-1:0], yticks=([10 * i for i in length(chess_result):-1:0], sorted_indices), xerror=sorted_confidence, title="Chess ranking (all)", label=false)

    savefig(chess_p, "$(comparison_folder_name)/chess_all_single_res.pdf")

    open("$(comparison_folder_name)/chess_all_single_res.tex", "w") do io
        println(io, tikz_chess((chess_result, confidence), collect(1:length(chess_result)), title="Chess ranking (all)"))
    end
end



if run_wilcoxon
    if !isdir("$(comparison_folder_name)/images")
        mkdir("$(comparison_folder_name)/images")
    end
    number_of_variations = length(comp_dfs)


    for graph_id in 1:l_df
        conf_mat = [
            pvalue(MannWhitneyUTest(Vector(df_i[graph_id, 2:end-4]), Vector(df_j[graph_id, 2:end-4]))) < 0.05 ?
            (mean(df_i[graph_id, :mean]) < mean(df_j[graph_id, :mean]) ? 1 : -1) : 0 for
            df_i in comp_dfs, df_j in comp_dfs
        ]
        conf_mat_d = [
            pvalue(MannWhitneyUTest(Vector(df_i[graph_id, 2:end-4]), Vector(df_j[graph_id, 2:end-4]))) < 0.05 ? 1 : 0 for
            df_i in comp_dfs, df_j in comp_dfs
        ]

        df = DataFrame(x=Int[], y=Float64[])

        for (i, df_i) in enumerate(comp_dfs)
            for v in df_i[graph_id, 2:end-4]
                push!(df, (i, v))
            end
        end


        point_sum += conf_mat
        point_sum_d += conf_mat_d

        if graph_id ∈ heatmaps_per_graph
            savefig(
                Plots.heatmap(
                    conf_mat_d,
                    yflip=true,
                    aspect_ratio=0.5,
                    color=[:white, :black],
                    showaxis=:xy,
                    xticks=1:number_of_variations,
                    yticks=1:number_of_variations,
                    title="$(comp_dfs[1][graph_id, 1])",
                ),
                "$(comparison_folder_name)/images/heat_$(comp_dfs[1][graph_id, 1]).pdf",
            )
            open("$(comparison_folder_name)/images/heat_$(comp_dfs[1][graph_id, 1]).tex", "w") do io
                println(io, tikz_heatmap(conf_mat_d, collect(1:number_of_variations)))
            end
            savefig(
                Plots.heatmap(
                    conf_mat,
                    yflip=true,
                    aspect_ratio=0.5,
                    showaxis=:xy,
                    xticks=1:number_of_variations,
                    yticks=1:number_of_variations,
                    title="$(comp_dfs[1][graph_id, 1])",
                ),
                "$(comparison_folder_name)/images/heat2_$(comp_dfs[1][graph_id, 1]).pdf",
            )
            open("$(comparison_folder_name)/images/heat2_$(comp_dfs[1][graph_id, 1]).tex", "w") do io
                println(io, tikz_heatmap(conf_mat, collect(1:number_of_variations)))
            end

            if (!any(x -> isnan(x), df[:, :x]) && !any(x -> isnan(x), df[:, :y]))
                savefig(
                    boxplot(
                        df[:, :x],
                        df[:, :y],
                        line=(2, :black),
                        fill=(0.3, :orange),
                        legend=false,
                        xticks=1:number_of_variations,
                        ylabel=f_name,
                        title="$(comp_dfs[1][graph_id, 1])",
                    ),
                    "$(comparison_folder_name)/images/box_$(comp_dfs[1][graph_id, 1]).pdf",
                )
                vx = Vector(df[:, :x])
                vy = Vector(df[:, :y])
                ys = [collect(vy[vx.==i]) for i in 1:maximum(df[:, :x])]
                open("$(comparison_folder_name)/images/box_$(comp_dfs[1][graph_id, 1]).tex", "w") do io
                    println(io, tikz_boxplot(ys, collect(1:number_of_variations), title="$(comp_dfs[1][graph_id, 1])"))
                end
            else
                println("No boxplot for $(f_name)_$(comp_dfs[1][graph_id, 1]) because of NaNs")
            end



            savefig(
                scatter(
                    df[:, :x],
                    df[:, :y],
                    legend=false,
                    xticks=1:number_of_variations,
                    title="$(comp_dfs[1][graph_id, 1])",
                ),
                "$(comparison_folder_name)/images/scatter_$(comp_dfs[1][graph_id, 1]).pdf",
            )

            df = DataFrame(conf_mat, ["$(i)" for i in 1:number_of_variations])
            df[!, "Config"] = ["$(i)" for i in 1:number_of_variations]
            CSV.write("$(comparison_folder_name)/images/$(comp_dfs[1][graph_id, 1]).csv", df)
        end
    end
    point_sum
    savefig(
        heatmap(
            point_sum,
            yflip=true,
            aspect_ratio=0.5,
            showaxis=:xy,
            xticks=1:number_of_variations,
            yticks=1:number_of_variations,
        ),
        "$(comparison_folder_name)/images/sum.pdf",
    )

    point_sum_d ./= l_df
    savefig(
        heatmap(
            point_sum_d,
            yflip=true,
            aspect_ratio=0.5,
            color=[:white, :black],
            showaxis=:xy,
            xticks=1:number_of_variations,
            yticks=1:number_of_variations,
        ),
        "$(comparison_folder_name)/images/percent.pdf",
    )

    sum_df = DataFrame(point_sum, ["$(i)" for i in 1:number_of_variations])
    sum_df[!, "Config"] = ["$(i)" for i in 1:number_of_variations]

    sum_of_rows = sum(point_sum, dims=2)
    best_version = argmax(sum_of_rows)[1]
    best_value = maximum(sum_of_rows)
    @show sum_of_rows
    sum_df[!, "Sum"] = sum_of_rows[:]

    CSV.write("$(comparison_folder_name)/images/sum.csv", sum_df)



end