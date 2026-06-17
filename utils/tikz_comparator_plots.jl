using Statistics

BOXPLOT_PREAMBLE = """
\\begin{tikzpicture}
\\begin{axis}[
    boxplot/draw direction=y,
    axis x line*=bottom,
    axis y line=left,
    enlarge y limits,
    ymajorgrids,"""

BOXPLOT_POST = """
\\end{axis}
\\end{tikzpicture} 
"""


function single_boxplot(lw, lq, m, uq, uw, outliers)
    outliers_str = join(outliers, " ")
    """
    \\addplot[
        blue,
        boxplot prepared={
            lower whisker=$(lw), lower quartile=$(lq),
            median=$(m),
            upper quartile=$(uq), upper whisker=$(uw),
        },
        mark=*,
    ] coordinates {$(outliers_str)};
    """
end

function tikz_boxplot(data, xlabels; whisker_range=1.5, title="")
    res = BOXPLOT_PREAMBLE

    res = """$(res)
    xtick={$(join(collect(1:length(data)),","))},
    xticklabels={$(join(xlabels, ","))},
    title={$(title)}
]
    """

    for values in data

        q1, q2, q3, q4, q5 = quantile(values, range(0, stop=1, length=5))
        outliers = []
        if Float64(whisker_range) != 0.0  # if the range is 0.0, the whiskers will extend to the data
            limit = whisker_range * (q4 - q2)
            inside = Float64[]
            for value in values
                if (value < (q2 - limit)) || (value > (q4 + limit))
                    push!(outliers, (0, value))
                else
                    push!(inside, value)
                end
            end
            # change q1 and q5 to show outliers
            # using maximum and minimum values inside the limits
            q1, q5 = extrema(inside)
            q1, q5 = (min(q1, q2), max(q4, q5)) # whiskers cannot be inside the box
        end

        res = """$(res)
        $(single_boxplot(q1, q2, q3, q4, q5, outliers))
        """
    end

    res = """$(res)
    $(BOXPLOT_POST)
    """

    res
end


HEATMAP_PREAMBLE = """
\\begin{tikzpicture}[scale=0.6]
  \\foreach \\y [count=\\n] in {"""

#=
      {74,25,39,20,3,3,3,3,3},
      {25,53,31,17,7,7,2,3,2},
      {39,31,37,24,3,3,3,3,3},
      {20,17,24,37,2,2,6,5,5},
      {3,7,3,2,12,1,0,0,0},
      {3,7,3,2,1,36,0,0,0},
      {3,2,3,6,0,0,45,1,1},
      {3,3,3,5,0,0,1,23,1},
      {3,2,3,5,0,0,1,1,78},
=#


function tikz_heatmap(data, labels; min_color="white", max_color="black", border_color="gray")
    res = HEATMAP_PREAMBLE

    max_data = maximum(data)
    data_norm = data ./ max_data * 100
    data_str = join(["{$(join(r, ","))}" for r in eachrow(data_norm)], ", \n")
    res = """$(res)
    $(data_str),
     } {
      % heatmap tiles
      \\foreach \\x [count=\\m] in \\y {
        \\node[draw=$(border_color), fill=$(max_color)!\\x!$(min_color), minimum size=6mm, text=white] at (\\m,-\\n) {\\x};
      }
    }

    % row labels
    \\foreach \\a [count=\\i] in {$(join(labels, ","))} {
        \\node[minimum size=6mm] at (0,-\\i) {\\a};
    }
    % column labels
    \\foreach \\a [count=\\i] in {$(join(labels, ","))} {
        \\node[minimum size=6mm] at (\\i, 0) {\\a};
    }
    \\end{tikzpicture}
    """

    res
end



function tikz_chess(data, labels; title="Chess comp")

    chess_result, confidence = data

    sorted_indices = sortperm(chess_result, rev=true)
    sorted_result = chess_result[sorted_indices]
    sorted_confidence = confidence[sorted_indices]

    res = """\\begin{tikzpicture}
\\begin{axis}[ 
    ytick={$(join(collect(1:length(chess_result)), ","))},
    yticklabels={$(join(labels[sorted_indices], ","))},
    ymajorgrids,
]
    \\addplot+ [
        only marks,
        title={$title},
        error bars/.cd,
            x dir=both,x explicit,
    ] coordinates {
    """

    l = length(sorted_indices)

    for (i, (avg, conf)) in enumerate(zip(sorted_result, sorted_confidence))
        res = """$(res)
            ($avg, $(l - i + 1))     +- ($(conf),0.0)"""
    end

    """$(res)
    };
\\end{axis}
\\end{tikzpicture} """
end
