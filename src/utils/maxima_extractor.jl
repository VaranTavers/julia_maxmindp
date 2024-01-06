using CSV
using DataFrames

# Extracts maxima from results_compiled files

function find_main_file(files)
  return "results_compiled.csv"
end

folders = readdir("results/")

summary_df_max = []
summary_df_mean = []
df_init = false

for folder in folders
  if isdir("results/$(folder)")
    files = readdir("results/$(folder)")

    main_file = find_main_file(files)

    if isfile("results/$(folder)/$(main_file)")
      df = CSV.read("results/$(folder)/$(main_file)", DataFrame)
      if !df_init
        global summary_df_max = df[:, ["graphs"]]
        global summary_df_mean = df[:, ["graphs"]]
        global df_init = true
      end
      summary_df_max[:, "$(folder)"] = df[:, "max"]
      summary_df_mean[:, "$(folder)"] = df[:, "mean"]
    end
  end
end

CSV.write("results/compile_max.csv", summary_df_max)
CSV.write("results/compile_mean.csv", summary_df_mean)