countmap(v) = countmap(identity, v)
function countmap(f, v)
    counts = Dict{eltype(f(first(v))),Int}()
    for x in v
        fx = f(x)
        counts[fx] = get!(counts, fx, 0) + 1
    end
    return counts
end

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
    Base.:^(t::NTuple{N}, p::PG.AbstractPermutation) where {N} =
        ntuple(i -> t[i^p], N)
    @testset "F4" begin
        F4 = let Σ = PG.PermGroup(PG.perm"(1,2,3,4)", PG.perm"(1,2)")
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
        @test PropertyT.Roots.classify_root_system(b, c, (false, true)) == :C₂

        long = F4[findfirst(r -> PropertyT.Roots.ℓ₂length(r) == sqrt(2), F4)]
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
            let Σ = PG.PermGroup(
                    PG.perm"(1,2,3,4,5,6,7,8)",
                    PG.perm"(1,2)",
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
                    p for p in Iterators.product(fill([-1, +1], 8)...) if
                    iseven(count(==(-1), p))
                )
                halfs = let x = (1, 1, 1, 1, 1, 1, 1, 1) .// 2
                    rts = unique(PropertyT.Roots.Root(x .* sgn) for sgn in signs)
                end

                union(long, halfs)
            end

        subtypes = Set([:A₂, Symbol("A₁×A₁")])

        @testset "E8" begin
            @test length(E8) == 240
            @test all(r -> PropertyT.Roots.ℓ₂length(r) ≈ sqrt(2), E8)

            let Ω = E8, α = first(Ω)
                counts = countmap([
                    PropertyT.Roots.classify_sub_root_system(Ω, α, γ) for
                    γ in Ω if !PropertyT.Roots.isproportional(α, γ)
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
                    PropertyT.Roots.classify_sub_root_system(Ω, α, γ) for
                    γ in Ω if !PropertyT.Roots.isproportional(α, γ)
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
                    PropertyT.Roots.classify_sub_root_system(Ω, α, γ) for
                    γ in Ω if !PropertyT.Roots.isproportional(α, γ)
                ])
                @test Set(keys(counts)) == subtypes
                d, r = divrem(counts[:A₂], 4)
                @info d, r
                @test r == 0 && d == 10
            end
        end
    end
end
