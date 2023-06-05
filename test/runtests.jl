using MaxMinDP_GA
using Test

# sumdpGreedyIN, sumdpRouletteIN, sumdpRandomIN, sumdpGreedyOUT, sumdpRouletteOUT, sumdpRandomOUT

@testset "sbts.jl_IN" begin
    @test begin
        vec = rand(1:100, 20)
        sumdpGreedyIN(vec) == argmin(vec)
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRouletteIN(vec; p=0.5) for _ in 1:numberOfRuns] .< 6) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRouletteIN(vec; p=0.2) for _ in 1:numberOfRuns] .< 3) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 20, 30, 40, 50, 10000, 10000, 10000, 10000, 10000]
        count([sumdpRouletteIN(vec; p=0.5) for _ in 1:numberOfRuns] .== 1) > numberOfRuns * 0.8
    end
    @test begin
        numberOfRuns = 50
        vec = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 1]
        count([sumdpRouletteIN(vec; p=0.7) for _ in 1:numberOfRuns] .== 10) > numberOfRuns * 0.9
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRandomIN(vec; p=0.5) for _ in 1:numberOfRuns] .< 6) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRandomIN(vec; p=0.2) for _ in 1:numberOfRuns] .< 3) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        count([sumdpRandomIN(vec; p=0.2) for _ in 1:numberOfRuns] .> 7) == numberOfRuns
    end
end

@testset "sbts.jl_OUT" begin
    @test begin
        vec = rand(1:100, 20)
        sumdpGreedyOUT(vec) == argmax(vec)
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRouletteOUT(vec; p=0.5) for _ in 1:numberOfRuns] .> 4) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRouletteOUT(vec; p=0.2) for _ in 1:numberOfRuns] .> 7) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [10000, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        count([sumdpRouletteOUT(vec; p=0.5) for _ in 1:numberOfRuns] .== 1) > numberOfRuns * 0.9
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 1, 1, 1, 1, 1, 1, 1, 1, 10000]
        count([sumdpRouletteOUT(vec; p=0.7) for _ in 1:numberOfRuns] .== 10) > numberOfRuns * 0.9
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRandomOUT(vec; p=0.5) for _ in 1:numberOfRuns] .> 4) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        count([sumdpRandomOUT(vec; p=0.2) for _ in 1:numberOfRuns] .> 7) == numberOfRuns
    end
    @test begin
        numberOfRuns = 50
        vec = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        count([sumdpRandomOUT(vec; p=0.2) for _ in 1:numberOfRuns] .< 3) == numberOfRuns
    end
end
