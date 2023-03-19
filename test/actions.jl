function test_action(basis, group, act)
    action = SymbolicWedderburn.action
    return @testset "action definition" begin
        @test all(basis) do b
            e = one(group)
            return action(act, e, b) == b
        end

        a = let a = rand(basis)
            while isone(a)
                a = rand(basis)
            end
            @assert !isone(a)
            a
        end

        g, h = let g_h = rand(group, 2)
            while any(isone, g_h)
                g_h = rand(group, 2)
            end
            @assert all(!isone, g_h)
            g_h
        end

        action = SymbolicWedderburn.action
        @test action(act, g, a) in basis
        @test action(act, h, a) in basis
        @test action(act, h, action(act, g, a)) == action(act, g * h, a)

        @test all([(g, h) for g in group for h in group]) do (g, h)
            x = action(act, h, action(act, g, a))
            y = action(act, g * h, a)
            return x == y
        end

        if act isa SymbolicWedderburn.ByPermutations
            @test all(basis) do b
                return action(act, g, b) ∈ basis && action(act, h, b) ∈ basis
            end
        end
    end
end

## Testing

@testset "Actions on SL(3,ℤ)" begin
    n = 3

    SL = MatrixGroups.SpecialLinearGroup{n}(Int8)
    RSL, S, sizes = PropertyT.group_algebra(SL; halfradius = 2)

    @testset "Permutation action" begin
        Γ = PermGroup(perm"(1,2)", Perm(circshift(1:n, -1)))
        ΓpA = PropertyT.action_by_conjugation(SL, Γ)

        test_action(basis(RSL), Γ, ΓpA)

        @testset "mps is successful" begin
            charsΓ =
                SymbolicWedderburn.Character{
                    Rational{Int},
                }.(SymbolicWedderburn.irreducible_characters(Γ))

            RΓ = SymbolicWedderburn._group_algebra(Γ)

            @time mps, ranks =
                SymbolicWedderburn.minimal_projection_system(charsΓ, RΓ)
            @test all(isone, ranks)
        end

        @testset "Wedderburn decomposition" begin
            wd = SymbolicWedderburn.WedderburnDecomposition(
                Rational{Int},
                Γ,
                ΓpA,
                basis(RSL),
                StarAlgebras.Basis{UInt16}(@view basis(RSL)[1:sizes[2]]),
            )

            @test length(invariant_vectors(wd)) == 918
            @test SymbolicWedderburn.size.(direct_summands(wd), 1) ==
                  [40, 23, 18]
            @test all(issimple, direct_summands(wd))
        end
    end

    @testset "Wreath action" begin
        Γ = let P = PermGroup(perm"(1,2)", Perm(circshift(1:n, -1)))
            PropertyT.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
        end

        ΓpA = PropertyT.action_by_conjugation(SL, Γ)

        test_action(basis(RSL), Γ, ΓpA)

        @testset "mps is successful" begin
            charsΓ =
                SymbolicWedderburn.Character{
                    Rational{Int},
                }.(SymbolicWedderburn.irreducible_characters(Γ))

            RΓ = SymbolicWedderburn._group_algebra(Γ)

            @time mps, ranks =
                SymbolicWedderburn.minimal_projection_system(charsΓ, RΓ)
            @test all(isone, ranks)
        end

        @testset "Wedderburn decomposition" begin
            wd = SymbolicWedderburn.WedderburnDecomposition(
                Rational{Int},
                Γ,
                ΓpA,
                basis(RSL),
                StarAlgebras.Basis{UInt16}(@view basis(RSL)[1:sizes[2]]),
            )

            @test length(invariant_vectors(wd)) == 247
            @test SymbolicWedderburn.size.(direct_summands(wd), 1) ==
                  [14, 9, 6, 14, 12]
            @test all(issimple, direct_summands(wd))
        end
    end
end

@testset "Actions on SAut(F4)" begin
    n = 4

    SAutFn = SpecialAutomorphismGroup(FreeGroup(n))
    RSAutFn, S, sizes = PropertyT.group_algebra(SAutFn; halfradius = 1)

    @testset "Permutation action" begin
        Γ = PermGroup(perm"(1,2)", Perm(circshift(1:n, -1)))
        ΓpA = PropertyT.action_by_conjugation(SAutFn, Γ)

        test_action(basis(RSAutFn), Γ, ΓpA)

        @testset "mps is successful" begin
            charsΓ =
                SymbolicWedderburn.Character{
                    Rational{Int},
                }.(SymbolicWedderburn.irreducible_characters(Γ))

            RΓ = SymbolicWedderburn._group_algebra(Γ)

            @time mps, ranks =
                SymbolicWedderburn.minimal_projection_system(charsΓ, RΓ)
            @test all(isone, ranks)
        end

        @testset "Wedderburn decomposition" begin
            wd = SymbolicWedderburn.WedderburnDecomposition(
                Rational{Int},
                Γ,
                ΓpA,
                basis(RSAutFn),
                StarAlgebras.Basis{UInt16}(@view basis(RSAutFn)[1:sizes[1]]),
            )

            @test length(invariant_vectors(wd)) == 93
            @test SymbolicWedderburn.size.(direct_summands(wd), 1) ==
                  [4, 8, 5, 4]
            @test all(issimple, direct_summands(wd))
        end
    end

    @testset "Wreath action" begin
        Γ = let P = PermGroup(perm"(1,2)", Perm(circshift(1:n, -1)))
            PropertyT.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
        end

        ΓpA = PropertyT.action_by_conjugation(SAutFn, Γ)

        test_action(basis(RSAutFn), Γ, ΓpA)

        @testset "mps is successful" begin
            charsΓ =
                SymbolicWedderburn.Character{
                    Rational{Int},
                }.(SymbolicWedderburn.irreducible_characters(Γ))

            RΓ = SymbolicWedderburn._group_algebra(Γ)

            @time mps, ranks =
                SymbolicWedderburn.minimal_projection_system(charsΓ, RΓ)
            @test all(isone, ranks)
        end

        @testset "Wedderburn decomposition" begin
            wd = SymbolicWedderburn.WedderburnDecomposition(
                Rational{Int},
                Γ,
                ΓpA,
                basis(RSAutFn),
                StarAlgebras.Basis{UInt16}(@view basis(RSAutFn)[1:sizes[1]]),
            )

            @test length(invariant_vectors(wd)) == 18
            @test SymbolicWedderburn.size.(direct_summands(wd), 1) ==
                  [1, 1, 2, 2, 1, 2, 2, 1]
            @test all(issimple, direct_summands(wd))
        end
    end
end
