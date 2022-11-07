@testset "Adj for SpN via grading" begin

    genus = 3
    halfradius = 2

    SpN = MatrixGroups.SymplecticGroup{2genus}(Int8)

    RSpN, S_sp, sizes_sp = PropertyT.group_algebra(SpN, halfradius=halfradius, twisted=true)

    Δ, Δs = let RG = RSpN, S = S_sp, ψ = identity
        Δ = RG(length(S)) - sum(RG(s) for s in S)
        Δs = PropertyT.laplacians(
            RG,
            S,
            x -> (gx = PropertyT.grading(ψ(x)); Set([gx, -gx])),
        )
        Δ, Δs
    end

    @testset "Adj correctness: genus=$genus" begin

        all_subtypes = (
            :A₁, :C₁, Symbol("A₁×A₁"), Symbol("C₁×C₁"), Symbol("A₁×C₁"), :A₂, :C₂
        )

        @test PropertyT.Adj(Δs, :A₂)[one(SpN)] == 384
        @test iszero(PropertyT.Adj(Δs, Symbol("A₁×A₁")))
        @test iszero(PropertyT.Adj(Δs, Symbol("C₁×C₁")))

        @testset "divisibility by 16" begin
            for subtype in all_subtypes
                subtype in (:A₁, :C₁) && continue
                @test isinteger(PropertyT.Adj(Δs, subtype)[one(SpN)] / 16)
            end
        end
        @test sum(PropertyT.Adj(Δs, subtype) for subtype in all_subtypes) == Δ^2
    end

end

