using JLD
using JuMP
import Primes: isprime
import SCS: SCSSolver
import Mosek: MosekSolver

using Mods

using Groups
using ProgressMeter


function SL_generatingset(n::Int)

    indexing = [(i,j) for i in 1:n for j in 1:n if i≠j]

    S = [E(i,j,N=n) for (i,j) in indexing];
    S = vcat(S, [convert(Array{Int,2},x') for x in S]);
    S = vcat(S, [convert(Array{Int,2},inv(x)) for x in S]);
    return unique(S)
end

function E(i::Int, j::Int; val=1, N::Int=3, mod=Inf)
   @assert i≠j
   m = eye(Int, N)
   m[i,j] = val
   if mod == Inf
       return m
   else
       return [Mod(x,mod) for x in m]
   end
end

function cofactor(i,j,M)
    z1 = ones(Bool,size(M,1))
    z1[i] = false

    z2 = ones(Bool,size(M,2))
    z2[j] = false

    return M[z1,z2]
end

import Base.LinAlg.det

function det(M::Array{Mod,2})
    if size(M,1) ≠ size(M,2)
        d = Mod(0,M[1,1].mod)
    elseif size(M,1) == 2
        d =  M[1,1]*M[2,2] - M[1,2]*M[2,1]
    else
        d = zero(eltype(M))
        for i in 1:size(M,1)
            d += (-1)^(i+1)*M[i,1]*det(cofactor(i,1,M))
        end
    end
#     @show (M, d)
    return d
end

function adjugate(M)
    K = similar(M)
    for i in 1:size(M,1), j in 1:size(M,2)
        K[j,i] = (-1)^(i+j)*det(cofactor(i,j,M))
    end
    return K
end

import Base: inv, one, zero, *

one(::Type{Mod}) = 1
zero(::Type{Mod}) = 0
zero(x::Mod) = Mod(x.mod)

function inv(M::Array{Mod,2})
    d = det(M)
    d ≠ 0*d || thow(ArgumentError("Matrix is not invertible!"))
    return inv(det(M))*adjugate(M)
    return adjugate(M)
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
include("property(T).jl")

main()
