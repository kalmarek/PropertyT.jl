using JuMP
import SCS: SCSSolver
import Mosek: MosekSolver

push!(LOAD_PATH, "./")

using GroupAlgebras
include("property(T).jl")


const VERBOSE=true


function E(i::Int, j::Int, N::Int=3)
    @assert i≠j
    k = eye(N)
    k[i,j] = 1
    return k
end

function SL_3ZZ_generating_set()
    S = [E(1,2), E(1,3), E(2,3)];
    S = vcat(S, [x' for x in S]);
    S = vcat(S, [inv(x) for x in S]);
    return S
end

const ID = eye(3)

const S₁ = SL_3ZZ_generating_set()


const TOL=10.0^-7
#solver = SCSSolver(eps=10.0^-TOL, max_iters=ITERATIONS, verbose=true);
solver = MosekSolver(MSK_DPAR_INTPNT_CO_TOL_REL_GAP=TOL,
#                      MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-15,
#                      MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-15,
#                      MSK_IPAR_PRESOLVE_USE=0,
                  QUIET=!VERBOSE)

# κ, A = solve_for_property_T(S₁, solver, verbose=VERBOSE)
# Δ, = prepare_Laplacian_and_constraints(S₁)

product_matrix = readdlm("SL3Z.product_matrix", Int)
L = readdlm("SL3Z.delta.coefficients")[:, 1]
Δ = GroupAlgebraElement(L, product_matrix)

A = readdlm("SL3Z.SDPmatrixA.Mosek")
κ = readdlm("SL3Z.kappa.Mosek")[1]

# @show eigvals(A)
@assert isapprox(eigvals(A), abs(eigvals(A)), atol=TOL)
@assert A == Symmetric(A)


const A_sqrt = real(sqrtm(A))

SOS_fp_diff, SOS_fp_L₁_distance = check_solution(κ, A_sqrt, Δ)

@show SOS_fp_L₁_distance
@show GroupAlgebras.ɛ(SOS_fp_diff)

κ_rational = rationalize(BigInt, κ;)
A_sqrt_rational = rationalize(BigInt, A_sqrt)
Δ_rational = rationalize(BigInt, Δ)

SOS_rational_diff, SOS_rat_L₁_distance = check_solution(κ_rational, A_sqrt_rational, Δ_rational)

@assert isa(SOS_rat_L₁_distance, Rational{BigInt})
@show float(SOS_rat_L₁_distance)
@show float(GroupAlgebras.ɛ(SOS_rational_diff))
