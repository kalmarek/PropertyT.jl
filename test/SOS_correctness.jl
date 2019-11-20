@testset "Correctness of HPC SOS computation" begin

    function prepare(G_name, λ, S_size)
        pm = load("$G_name/delta.jld", "pm")
        P = load("$G_name/$λ/solution.jld", "P")
        @time Q = real(sqrt(P))

        Δ_coeff = SparseVector(maximum(pm), collect(1:1+S_size), [S_size; ((-1.0) for i in 1:S_size)...])

        Δ²_coeff = GroupRings.GRmul!(spzeros(length(Δ_coeff)), Δ_coeff, Δ_coeff, pm)

        eoi = Δ²_coeff - λ*Δ_coeff

        Q = PropertyT.augIdproj(Q)

        return eoi, pm, Q
    end

    #########################################################
    NAME = "SL(3,Z)"
    eoi, pm, Q = prepare(NAME, 0.1, 3*2*2)

    @time sos_sqr = PropertyT.compute_SOS_square(pm, Q)
    @time sos_hpc = PropertyT.compute_SOS(pm, Q)

    @test norm(sos_sqr - sos_hpc, 1) < 3e-12
    @info "$NAME:\nDifference in l₁-norm between square and hpc sos decompositions:" norm(eoi-sos_sqr,1) norm(eoi-sos_hpc,1) norm(sos_sqr - sos_hpc, 1)

    #########################################################
    NAME = "SL(3,Z)_orbit"
    eoi, pm, Q = prepare(NAME, 0.27, 3*2*2)

    @time sos_sqr = PropertyT.compute_SOS_square(pm, Q)
    @time sos_hpc = PropertyT.compute_SOS(pm, Q)

    @test norm(sos_sqr - sos_hpc, 1) < 5e-12
    @info "$NAME:\nDifference in l₁-norm between square and hpc sos decompositions:" norm(eoi-sos_sqr,1) norm(eoi-sos_hpc,1) norm(sos_sqr - sos_hpc, 1)

    #########################################################
    NAME = "SL(4,Z)_orbit"
    eoi, pm, Q = prepare(NAME, 1.3, 4*3*2)

    @time sos_sqr = PropertyT.compute_SOS_square(pm, Q)
    @time sos_hpc = PropertyT.compute_SOS(pm, Q)

    @test norm(sos_sqr - sos_hpc, 1) < 2e-10
    @info "$NAME:\nDifference in l₁-norm between square and hpc sos decompositions:" norm(eoi-sos_sqr,1) norm(eoi-sos_hpc,1) norm(sos_sqr - sos_hpc, 1)

    #########################################################
    NAME = "SAut(F3)_orbit"
    eoi, pm, Q = prepare(NAME, 0.15, 4*3*2*2)

    @time sos_sqr = PropertyT.compute_SOS_square(pm, Q)
    @time sos_hpc = PropertyT.compute_SOS(pm, Q)

    @test norm(sos_sqr - sos_hpc, 1) < 6e-11
    @info "$NAME:\nDifference in l₁-norm between square and hpc sos decompositions:" norm(eoi-sos_sqr,1) norm(eoi-sos_hpc,1) norm(sos_sqr - sos_hpc, 1)

end
