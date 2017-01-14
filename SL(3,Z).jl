using JuMP
import SCS: SCSSolver
import Mosek: MosekSolver

workers_processes = addprocs()

@everywhere push!(LOAD_PATH, "./")
using GroupAlgebras
@everywhere include("property(T).jl")

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
# const VERBOSE=true
#solver = SCSSolver(eps=TOL, max_iters=ITERATIONS, verbose=VERBOSE);
# solver = MosekSolver(MSK_DPAR_INTPNT_CO_TOL_REL_GAP=TOL,
# #                      MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-15,
# #                      MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-15,
# #                      MSK_IPAR_PRESOLVE_USE=0,
#                   QUIET=!VERBOSE)

# κ, A = solve_for_property_T(S₁, solver, verbose=VERBOSE)


const product_matrix = readdlm("SL3Z.product_matrix", Int)
const L = readdlm("SL3Z.delta.coefficients")[:, 1]
const Δ = GroupAlgebraElement(L, product_matrix)

const A = readdlm("SL3Z.SDPmatrixA.Mosek")
const κ = readdlm("SL3Z.kappa.Mosek")[1]

@assert isapprox(eigvals(A), abs(eigvals(A)), atol=TOL)
@assert A == Symmetric(A)

const A_sqrt = real(sqrtm(A))

const SOS_fp_diff, SOS_fp_L₁_distance = check_solution(κ, A_sqrt, Δ)

@show SOS_fp_L₁_distance
@show GroupAlgebras.ɛ(SOS_fp_diff)

const κ_rational = rationalize(BigInt, κ, tol=TOL)
const A_sqrt_rational = rationalize(BigInt, A_sqrt, tol=TOL)
const Δ_rational = rationalize(BigInt, Δ, tol=TOL)

const SOS_rational_diff, SOS_rat_L₁_distance = check_solution(κ_rational, A_sqrt_rational, Δ_rational)

@assert isa(SOS_rat_L₁_distance, Rational{BigInt})
@show float(SOS_rat_L₁_distance)
@show float(GroupAlgebras.ɛ(SOS_rational_diff))

const A_sqrt_augmented = correct_to_augmentation_ideal(A_sqrt_rational)

const SOS_rational_aug_diff, SOS_aug_rat_L₁_distance = check_solution(κ_rational, A_sqrt_augmented, Δ_rational)

@assert isa(SOS_aug_rat_L₁_distance, Rational{BigInt})
@assert GroupAlgebras.ɛ(SOS_rational_aug_diff) == 0//1

@show float(SOS_aug_rat_L₁_distance)
@show float(κ_rational - 2^3*SOS_aug_rat_L₁_distance)

rmprocs(workers_processes)
