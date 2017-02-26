using JLD
using JuMP
import SCS: SCSSolver
import Mosek: MosekSolver

using Groups
using ProgressMeter


function SL₃ℤ_generatingset()

    function E(i::Int, j::Int, N::Int=3)
        @assert i≠j
        k = eye(N)
        k[i,j] = 1
        return k
    end

    S = [E(1,2), E(1,3), E(2,3)];
    S = vcat(S, [x' for x in S]);
    S = vcat(S, [inv(x) for x in S]);
    return S
end

function prepare_Δ_sdp_constraints(identity, S)
    @show length(S)

    B₁ = vcat([identity], S)
    B₂ = products(B₁, B₁);
    B₃ = products(B₁, B₂);
    B₄ = products(B₁, B₃);
    @assert B₄[1:length(B₂)] == B₂

    product_matrix = create_product_matrix(B₄,length(B₂));
    sdp_constraints = constraints_from_pm(product_matrix, length(B₄))
    L_coeff = splaplacian_coeff(S, B₂, length(B₄));
    Δ = GroupAlgebraElement(L_coeff, product_matrix)

    return Δ, sdp_constraints
end

function load_Δ_sdp_constraints(name::String;cached=true)
    pm_filename = "$name.product_matrix.jld"
    Δ_coeff_filename = "$name.delta.coefficients.jld"
    f₁ = isfile(pm_filename)
    f₂ = isfile(Δ_coeff_filename)
    if cached && f₁ && f₂
        println("Loading precomputed pm, Δ, sdp_constraints...")
        product_matrix = load(pm_filename, "pm")
        L = load(Δ_coeff_filename, "Δ")[:, 1]
        Δ = GroupAlgebraElement(L, Array{Int,2}(product_matrix))
        sdp_constraints = constraints_from_pm(product_matrix)
    else
        println("Computing pm, Δ, sdp_constraints...")
        ID = eye(Int, 3)
        S = SL₃ℤ_generatingset()
        Δ, sdp_constraints = prepare_Δ_sdp_constraints(ID, S)

        save(pm_filename, "pm", Δ.product_matrix)
        save(Δ_coeff_filename, "Δ", Δ.coefficients)

    end
    return Δ, sdp_constraints
end


function compute_κ_A(name::String, Δ, sdp_constraints;
    cached = true,
    tol = 1e-7,
    verbose = false,
    # solver = MosekSolver(INTPNT_CO_TOL_REL_GAP=tol, QUIET=!verbose))
    solver = SCSSolver(eps=tol, max_iters=20000, cg_rate=3, verbose=verbose))

    f₁ = isfile("$name.kappa")
    f₂ = isfile("$name.SDPmatrixA")

    if cached && f₁ && f₂
        println("Loading precomputed κ, A...")
        A = readdlm("$name.SDPmatrixA")
        κ = readdlm("$name.kappa")[1]
    else
        println("Solving SDP problem maximizing κ...")
        κ, A = solve_SDP(sdp_constraints, Δ, solver, verbose=verbose)
        # writedlm("$name.kappa", kappa)
        # writedlm("$name.SDPmatrixA", A)
    end
    return κ, A
end

function main()
    const NAME = "SL3Z"
    const VERBOSE = true
    const TOL=1e-7
    const Δ, sdp_constraints = load_Δ_sdp_constraints(NAME)
    const κ, A = compute_κ_A(NAME, Δ, sdp_constraints, cached=false, verbose=VERBOSE)

    if maximum(A) < 1e-2
        warn("Solver might not solved the problem successfully and the positive solution is due to floating-point error, proceeding anyway...")
    end

    if κ > 0
        @assert A == Symmetric(A)
        const A_sqrt = real(sqrtm(A))

        T = ℚ_distance_to_positive_cone(Δ, κ, A, tol=TOL, verbose=VERBOSE)

        if T < 0
            println("$NAME HAS property (T)!")
        else
            println("$NAME may NOT HAVE property (T)!")
        end

    else
        println("$κ < 0: $NAME may NOT HAVE property (T)!")
    end
end

@everywhere push!(LOAD_PATH, "./")
using GroupAlgebras
@everywhere include("property(T).jl")

main()
