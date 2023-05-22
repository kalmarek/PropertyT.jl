countmap(v) = countmap(identity, v)
function countmap(f, v)
    counts = Dict{eltype(f(first(v))),Int}()
    for x in v
        fx = f(x)
        counts[fx] = get!(counts, fx, 0) + 1
    end
    return counts
end

@testset "Chevalley" begin
    @testset "classify_root_system" begin
        α = PropertyT.Roots.Root([1, -1, 0])
        β = PropertyT.Roots.Root([0, 1, -1])
        γ = PropertyT.Roots.Root([2, 0, 0])

        @test PropertyT.Roots.classify_root_system(α, β, (false, false)) == :A₂
        @test PropertyT.Roots.classify_root_system(α, γ, (false, true)) == :C₂
        @test PropertyT.Roots.classify_root_system(β, γ, (false, true)) ==
              Symbol("A₁×C₁")
    end

    @testset "Exceptional root systems" begin
        @testset "F4" begin
            F4 =
                let Σ = PermutationGroups.PermGroup(
                        perm"(1,2,3,4)",
                        perm"(1,2)",
                    )
                    long = let x = (1, 1, 0, 0) .// 1
                        PropertyT.Roots.Root.(
                            union(
                                (x^g for g in Σ),
                                ((x .* (-1, 1, 1, 1))^g for g in Σ),
                                ((-1 .* x)^g for g in Σ),
                            ),
                        )
                    end

                    short = let x = (1, 0, 0, 0) .// 1
                        PropertyT.Roots.Root.(
                            union((x^g for g in Σ), ((-1 .* x)^g for g in Σ))
                        )
                    end

                    signs = collect(Iterators.product(fill([-1, +1], 4)...))
                    halfs = let x = (1, 1, 1, 1) .// 2
                        PropertyT.Roots.Root.(union(x .* sgn for sgn in signs))
                    end

                    union(long, short, halfs)
                end

            @test length(F4) == 48

            a = F4[1]
            @test isapprox(PropertyT.Roots.ℓ₂length(a), sqrt(2))
            b = F4[6]
            @test isapprox(PropertyT.Roots.ℓ₂length(b), sqrt(2))
            c = a + b
            @test isapprox(PropertyT.Roots.ℓ₂length(c), 2.0)
            @test PropertyT.Roots.classify_root_system(b, c, (false, true)) ==
                  :C₂

            long =
                F4[findfirst(r -> PropertyT.Roots.ℓ₂length(r) == sqrt(2), F4)]
            short = F4[findfirst(r -> PropertyT.Roots.ℓ₂length(r) == 1.0, F4)]

            subtypes = Set([:C₂, :A₂, Symbol("A₁×C₁")])

            let Ω = F4, α = long
                counts = countmap([
                    PropertyT.Roots.classify_sub_root_system(Ω, α, γ) for
                    γ in Ω if !PropertyT.Roots.isproportional(α, γ)
                ])
                @test Set(keys(counts)) == subtypes
                d, r = divrem(counts[:C₂], 6)
                @test r == 0 && d == 3

                d, r = divrem(counts[:A₂], 4)
                @test r == 0 && d == 4
            end

            let Ω = F4, α = short
                counts = countmap([
                    PropertyT.Roots.classify_sub_root_system(Ω, α, γ) for
                    γ in Ω if !PropertyT.Roots.isproportional(α, γ)
                ])
                @test Set(keys(counts)) == subtypes
                d, r = divrem(counts[:C₂], 6)
                @test r == 0 && d == 3

                d, r = divrem(counts[:A₂], 4)
                @test r == 0 && d == 4
            end
        end

        @testset "E6-7-8 exceptional root systems" begin
            E8 =
                let Σ = PermutationGroups.PermGroup(
                        perm"(1,2,3,4,5,6,7,8)",
                        perm"(1,2)",
                    )
                    long = let x = (1, 1, 0, 0, 0, 0, 0, 0) .// 1
                        PropertyT.Roots.Root.(
                            union(
                                (x^g for g in Σ),
                                ((x .* (-1, 1, 1, 1, 1, 1, 1, 1))^g for g in Σ),
                                ((-1 .* x)^g for g in Σ),
                            ),
                        )
                    end

                    signs = collect(
                        p for
                        p in Iterators.product(fill([-1, +1], 8)...) if
                        iseven(count(==(-1), p))
                    )
                    halfs = let x = (1, 1, 1, 1, 1, 1, 1, 1) .// 2
                        rts = unique(
                            PropertyT.Roots.Root(x .* sgn) for sgn in signs
                        )
                    end

                    union(long, halfs)
                end

            subtypes = Set([:A₂, Symbol("A₁×A₁")])

            @testset "E8" begin
                @test length(E8) == 240
                @test all(r -> PropertyT.Roots.ℓ₂length(r) ≈ sqrt(2), E8)

                let Ω = E8, α = first(Ω)
                    counts = countmap([
                        PropertyT.Roots.classify_sub_root_system(Ω, α, γ)
                        for γ in Ω if !PropertyT.Roots.isproportional(α, γ)
                    ])
                    @test Set(keys(counts)) == subtypes
                    d, r = divrem(counts[:A₂], 4)
                    @test r == 0 && d == 28
                end
            end
            @testset "E7" begin
                E7 = filter(r -> iszero(sum(r.coord)), E8)
                @test length(E7) == 126

                let Ω = E7, α = first(Ω)
                    counts = countmap([
                        PropertyT.Roots.classify_sub_root_system(Ω, α, γ)
                        for γ in Ω if !PropertyT.Roots.isproportional(α, γ)
                    ])
                    @test Set(keys(counts)) == subtypes
                    d, r = divrem(counts[:A₂], 4)
                    @test r == 0 && d == 16
                end
            end

            @testset "E6" begin
                E6 = filter(
                    r -> r.coord[end] == r.coord[end-1] == r.coord[end-2],
                    E8,
                )
                @test length(E6) == 72

                let Ω = E6, α = first(Ω)
                    counts = countmap([
                        PropertyT.Roots.classify_sub_root_system(Ω, α, γ)
                        for γ in Ω if !PropertyT.Roots.isproportional(α, γ)
                    ])
                    @test Set(keys(counts)) == subtypes
                    d, r = divrem(counts[:A₂], 4)
                    @info d, r
                    @test r == 0 && d == 10
                end
            end
        end
    end

    @testset "Levels in Sp2n" begin
        function level(rootsystem, level::Integer)
            1 ≤ level ≤ 4 || throw("level is implemented only for i ∈{1,2,3,4}")
            level == 1 && return PropertyT.Adj(rootsystem, :C₁) # always positive
            level == 2 && return PropertyT.Adj(rootsystem, :A₁) +
                   PropertyT.Adj(rootsystem, Symbol("C₁×C₁")) +
                   PropertyT.Adj(rootsystem, :C₂) # C₂ is not positive
            level == 3 && return PropertyT.Adj(rootsystem, :A₂) +
                   PropertyT.Adj(rootsystem, Symbol("A₁×C₁"))
            level == 4 && return PropertyT.Adj(rootsystem, Symbol("A₁×A₁")) # positive
        end

        n = 5
        G = MatrixGroups.SymplecticGroup{2n}(Int8)
        RG, S, sizes = PropertyT.group_algebra(G; halfradius = 1)

        Weyl = let N = n
            P = PermGroup(perm"(1,2)", Perm(circshift(1:N, -1)))
            Groups.Constructions.WreathProduct(PermGroup(perm"(1,2)"), P)
        end
        act = PropertyT.action_by_conjugation(G, Weyl)

        function ^ᵃ(x, w::Groups.Constructions.WreathProductElement)
            return SymbolicWedderburn.action(act, w, x)
        end

        Sₙ = S
        Δsₙ = PropertyT.laplacians(
            RG,
            Sₙ,
            x -> (gx = PropertyT.grading(x); Set([gx, -gx])),
        )

        function natural_embedding(i, Sp2m, Sp2n)
            _dim(::MatrixGroups.ElementarySymplectic{N}) where {N} = N
            n = _dim(first(alphabet(Sp2n))) ÷ 2
            m = _dim(first(alphabet(Sp2m))) ÷ 2
            l = alphabet(Sp2m)[i]
            i, j = if l.symbol === :A
                l.i, l.j
            elseif l.symbol === :B
                ifelse(l.i ≤ m, (l.i, l.j - m + n), (l.i - m + n, l.j))
            else
                throw("unknown type: $(l.symbol)")
            end
            image_of_l =
                MatrixGroups.ElementarySymplectic{2n}(l.symbol, i, j, l.val)
            return Groups.word_type(Sp2n)([alphabet(Sp2n)[image_of_l]])
        end

        @testset "Sp4 ↪ Sp12" begin
            m = 2
            Sₘ = let m = m, Sp2n = G
                Sp2m = MatrixGroups.SymplecticGroup{2m}(Int8)
                h = Groups.Homomorphism(
                    natural_embedding,
                    Sp2m,
                    Sp2n;
                    check = false,
                )
                S = h.(gens(Sp2m))
                S = union!(S, inv.(S))
            end

            Δsₘ = PropertyT.laplacians(
                RG,
                Sₘ,
                x -> (gx = PropertyT.grading(x); Set([gx, -gx])),
            )

            function k(n, m, i)
                return 2^n * factorial(m) * factorial(n - i) ÷ factorial(m - i)
            end

            @testset "Level $i" for i in 1:4
                Levᵢᵐ = level(Δsₘ, i)
                Levᵢⁿ = level(Δsₙ, i)

                if 1 ≤ i ≤ 2
                    @test !iszero(Levᵢᵐ)
                    @time Σ_W_Levᵢᵐ = sum(Levᵢᵐ^ᵃw for w in Weyl)

                    @test isinteger(Σ_W_Levᵢᵐ[one(G)] / Levᵢⁿ[one(G)])
                    @test Σ_W_Levᵢᵐ[one(G)] / Levᵢⁿ[one(G)] == k(n, m, i)
                    @test Σ_W_Levᵢᵐ == k(n, m, i) * Levᵢⁿ
                else
                    @test iszero(Levᵢᵐ)
                    @test !iszero(Levᵢⁿ)
                end
            end
        end

        @testset "Sp8 ↪ Sp12" begin
            m = 4
            Sₘ = let m = m, Sp2n = G
                Sp2m = MatrixGroups.SymplecticGroup{2m}(Int8)
                h = Groups.Homomorphism(
                    natural_embedding,
                    Sp2m,
                    Sp2n;
                    check = false,
                )
                S = h.(gens(Sp2m))
                S = union!(S, inv.(S))
            end

            Δsₘ = PropertyT.laplacians(
                RG,
                Sₘ,
                x -> (gx = PropertyT.grading(x); Set([gx, -gx])),
            )

            function k(n, m, i)
                return 2^n * factorial(m) * factorial(n - i) ÷ factorial(m - i)
            end

            @testset "Level $i" for i in 1:4
                Levᵢᵐ = level(Δsₘ, i)
                Levᵢⁿ = level(Δsₙ, i)

                @test !iszero(Levᵢᵐ)
                @time Σ_W_Levᵢᵐ = sum(Levᵢᵐ^ᵃw for w in Weyl)

                @test isinteger(Σ_W_Levᵢᵐ[one(G)] / Levᵢⁿ[one(G)])
                @test Σ_W_Levᵢᵐ[one(G)] / Levᵢⁿ[one(G)] == k(n, m, i)
                @test Σ_W_Levᵢᵐ == k(n, m, i) * Levᵢⁿ
            end
        end
    end
end
