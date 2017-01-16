using Combinatorics

using JuMP
import SCS: SCSSolver
import Mosek: MosekSolver

push!(LOAD_PATH, "./")
using SemiDirectProduct
using GroupAlgebras
include("property(T).jl")

const N = 4

const VERBOSE = true

function permutation_matrix(p::Vector{Int})
    n = length(p)
    sort(p) == collect(1:n) || throw(ArgumentError("Input array must be a permutation of 1:n"))
    A = eye(n)
    return A[p,:]
end

SymmetricGroup(n) = [nthperm(collect(1:n), k) for k in 1:factorial(n)]

# const SymmetricGroup = [permutation_matrix(x) for x in SymmetricGroup_perms]

function E(i, j; dim::Int=N)
    @assert i≠j
    k = eye(dim)
    k[i,j] = 1
    return k
end

function eltary_basis_vector(i; dim::Int=N)
    result = zeros(dim)
    if 0 < i ≤ dim
        result[i] = 1
    end
    return result
end

v(i; dim=N) = eltary_basis_vector(i,dim=dim)

ϱ(i,j::Int,n=N) = SemiDirectProductElement(E(i,j,dim=n), v(j,dim=n))
λ(i,j::Int,n=N) = SemiDirectProductElement(E(i,j,dim=n), -v(j,dim=n))

function ɛ(i, n::Int=N)
    result = eye(n)
    result[i,i] = -1
    return SemiDirectProductElement(result)
end

σ(permutation::Vector{Int}) =
     SemiDirectProductElement(permutation_matrix(permutation))

# Standard generating set: 103 elements

function generatingset_ofAutF(n::Int=N)
    indexing = [[i,j] for i in 1:n for j in 1:n if i≠j]
    ϱs = [ϱ(ij...) for ij in indexing]
    λs = [λ(ij...) for ij in indexing]
    ɛs = [ɛ(i) for i in 1:N]
    σs = [σ(perm) for perm in SymmetricGroup(n)]
    S = vcat(ϱs, λs, ɛs, σs);
    S = unique(vcat(S, [inv(x) for x in S]));
    return S
end

const ID = eye(N+1)

const S₁ = generatingset_ofAutF(N)

matrix_S₁ = [matrix_repr(x) for x in S₁]

const TOL=10.0^-7

matrix_S₁[1:10,:][:,1]

Δ, cm = prepare_Laplacian_and_constraints(matrix_S₁)

#solver = SCSSolver(eps=TOL, max_iters=ITERATIONS, verbose=true);
solver = MosekSolver(MSK_DPAR_INTPNT_CO_TOL_REL_GAP=TOL,
#                      MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-15,
#                      MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-15,
#                      MSK_IPAR_PRESOLVE_USE=0,
                  QUIET=!VERBOSE)

# κ, A = solve_for_property_T(S₁, solver, verbose=VERBOSE)

product_matrix = readdlm("SL₃Z.product_matrix", Int)
L = readdlm("SL₃Z.Δ.coefficients")[:, 1]
Δ = GroupAlgebraElement(L, product_matrix)

A = readdlm("matrix.A.Mosek")
κ = readdlm("kappa.Mosek")[1]

# @show eigvals(A)
@assert isapprox(eigvals(A), abs(eigvals(A)), atol=TOL)
@assert A == Symmetric(A)


const A_sqrt = real(sqrtm(A))

SOS_EOI_fp_L₁, Ω_fp_dist = check_solution(κ, A_sqrt, Δ)

κ_rational = rationalize(BigInt, κ;)
A_sqrt_rational = rationalize(BigInt, A_sqrt)
Δ_rational = rationalize(BigInt, Δ)

SOS_EOI_rat_L₁, Ω_rat_dist = check_solution(κ_rational, A_sqrt_rational, Δ_rational)
