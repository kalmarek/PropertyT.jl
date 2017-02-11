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

function generate_B₂_and_B₄(B₁)
    B₂ = products(B₁, B₁);
    B₃ = products(B₁, B₂);
    B₄ = products(B₁, B₃);

    @assert B₄[1:length(B₂)] == B₂
    return B₂, B₄;
end

function prepare_Laplacian_and_constraints(S, identity)

    B₂, B₄ = generate_B₂_and_B₄(vcat([identity], S))
    product_matrix = create_product_matrix(B₄,length(B₂));
    sdp_constraints = constraints_from_pm(product_matrix, length(B₄))
    L_coeff = splaplacian_coeff(S, B₄);

    return GroupAlgebraElement(L_coeff, product_matrix), sdp_constraints
end

function prepare_Δ_sdp_constraints(name::String;cached=true)
    f₁ = isfile("$name.product_matrix")
    f₂ = isfile("$name.delta.coefficients")
    if cached && f₁ && f₂
        println("Loading precomputed pm, Δ, sdp_constraints...")
        product_matrix = readdlm("$name.product_matrix", Int)
        L = readdlm("$name.delta.coefficients")[:, 1]
        Δ = GroupAlgebraElement(L, product_matrix)
        sdp_constraints = constraints_from_pm(product_matrix)
    else
        println("Computing pm, Δ, sdp_constraints...")
        ID = eye(Int, 3)
        S₁ = SL₃ℤ_generatingset()
        Δ, sdp_constraints = prepare_Laplacian_and_constraints(S₁, ID)
        writedlm("$name.delta.coefficients", Δ.coefficients)
        writedlm("$name.product_matrix", Δ.product_matrix)
    end
    return Δ, sdp_constraints
end


function compute_κ_A(name::String, Δ, sdp_constraints;
    cached = true,
    tol = TOL,
    verbose = VERBOSE,
    solver = MosekSolver(INTPNT_CO_TOL_REL_GAP=tol, QUIET=!verbose))
    # solver = SCSSolver(eps=TOL, max_iters=ITERATIONS, verbose=VERBOSE))

    f₁ = isfile("$name.kappa")
    f₂ = isfile("$name.SDPmatrixA")

    if cached && f₁ && f₂
        println("Loading precomputed κ, A...")
        A = readdlm("$name.SDPmatrixA")
        κ = readdlm("$name.kappa")[1]
    else
        println("Solving SDP problem maximizing κ...")
        κ, A = solve_SDP(sdp_constraints, Δ, solver, verbose=verbose)
        writedlm("$name.kappa", kappa)
        writedlm("$name.SDPmatrixA", A)
    end
    return κ, A
end


workers_processes = addprocs()
@everywhere push!(LOAD_PATH, "./")
using GroupAlgebras
@everywhere include("property(T).jl")

const NAME = "SL3Z"
const VERBOSE = true
const TOL=1e-7
const Δ, sdp_constraints = prepare_Δ_sdp_constraints(NAME)
const κ, A = compute_κ_A(NAME, Δ, sdp_constraints)

if κ > 0
    @time T = ℚ_distance_to_positive_cone(Δ, κ, A, tol=TOL, verbose=VERBOSE)

    if T < 0
        println("$NAME HAS property (T)!")
    else
        println("$NAME may NOT HAVE property (T)!")
    end

else
    println("$κ < 0: $NAME may NOT HAVE property (T)!")
end

rmprocs(workers_processes)
