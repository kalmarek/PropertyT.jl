@testset "Adj via grading" begin

    @testset "SL(n,Z) & Aut(F₄)" begin
        n = 4
        halfradius = 1
        SL = MatrixGroups.SpecialLinearGroup{n}(Int8)
        RSL, S, sizes = PropertyT.group_algebra(SL, halfradius=halfradius, twisted=true)

        Δ = RSL(length(S)) - sum(RSL(s) for s in S)

        Δs = let ψ = identity
            PropertyT.laplacians(
                RSL,
                S,
                x -> (gx = PropertyT.grading(ψ(x)); Set([gx, -gx])),
            )
        end

        sq, adj, op = PropertyT.SqAdjOp(RSL, n)

        @test PropertyT.Adj(Δs, :A₁) == sq
        @test PropertyT.Adj(Δs, :A₂) == adj
        @test PropertyT.Adj(Δs, Symbol("A₁×A₁")) == op


        halfradius = 1
        G = SpecialAutomorphismGroup(FreeGroup(n))
        RG, S, sizes = PropertyT.group_algebra(G, halfradius=halfradius, twisted=true)

        Δ = RG(length(S)) - sum(RG(s) for s in S)

        Δs = let ψ = Groups.Homomorphism(Groups._abelianize, G, SL)
            PropertyT.laplacians(
                RG,
                S,
                x -> (gx = PropertyT.grading(ψ(x)); Set([gx, -gx])),
            )
        end

        sq, adj, op = PropertyT.SqAdjOp(RG, n)

        @test PropertyT.Adj(Δs, :A₁) == sq
        @test PropertyT.Adj(Δs, :A₂) == adj
        @test PropertyT.Adj(Δs, Symbol("A₁×A₁")) == op
    end


    @testset "Symplectic group" begin
        @testset "Sp2(ℤ)" begin
            genus = 2
            halfradius = 1

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

            sq = sum(Δᵢ^2 for Δᵢ in values(Δs))
            @test PropertyT.Adj(Δs, :C₂) + sq == Δ^2
        end

        genus = 3
        halfradius = 1

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

        @testset "Adj numerics for genus=$genus" begin

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
end

