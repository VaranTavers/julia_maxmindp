module Exact

using ..DispersionProblems

using Statistics

function exact_bt(k, vals, n, m, minDists, maxVals, maxFit; fitF=DispersionProblems.calculateMinSumdp)
    last = 0

    if k > m
        return vals, fitF(vals, minDists)
    end

    if k != 1
        last = vals[k-1]
    end

    for i in last+1:n
        vals[k] = i
        iVals, iFit = exact_bt(k + 1, vals, n, m, minDists, maxVals, maxFit; fitF=fitF)
        if iFit > maxFit
            maxVals = deepcopy(iVals)
            maxFit = iFit
        end
    end

    maxVals, maxFit
end

function exact(minDists, m; fitF=DispersionProblems.calcMinSumdp)
    n, _ = size(minDists)
    exact_bt(1, zeros(Int64, m), n, m, minDists, [0, 0, 0, 0, 0], 0; fitF=fitF)
end

end