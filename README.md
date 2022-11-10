# Property(T)

[![CI](https://github.com/kalmarek/PropertyT.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/kalmarek/PropertyT.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/kalmarek/PropertyT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kalmarek/PropertyT.jl)

This package is concerned with sum of squares decompositions in group rings of finitely presented groups.
Please have a look into e.g. this [test](https://github.com/kalmarek/PropertyT.jl/blob/master/test/1712.07167.jl#L87) to see how this package can be used to prove Kazdhan Property (T) for a finitely presented group. For an example applications have a look at our papers:
* M. Kaluba and P.W. Nowak _Certifying numerical estimates for spectral gaps_ [1703.09680](https://arxiv.org/abs/1703.09680)
* M. Kaluba, P.W. Nowak and N. Ozawa *$Aut(F₅)$ has property (T)* [1712.07167](https://arxiv.org/abs/1712.07167), and
* M. Kaluba, D. Kielak and P.W. Nowak *On property (T) for $Aut(Fₙ)$ and $SLₙ(Z)$* [1812.03456](https://arxiv.org/abs/1812.03456).

The package depends on
 * [`Groups.jl`](https://github.com/kalmarek/Groups.jl) for computations with finitely presented groups,
 * [`SymbolicWedderburn.jl`](https://github.com/kalmarek/SymbolicWedderburn.jl) for symmetrizing the sum of squares relaxations of positivity problems,
 * [`JuMP.jl`](https://github.com/JuliaOpt/JuMP.jl) for formulating the optimization problems, and
 * [`SCS.jl`](https://github.com/JuliaOpt/SCS.jl) wrapper for the [`scs` solver](https://github.com/cvxgrp/scs), or
 * [`COSMO.jl`](https://github.com/oxfordcontrol/COSMO.jl) solver to solve the problems.

 Certification of the results is done via `ℓ₁`-convexity of the sum-of-squares cone and our knowledge of its interior points. The certified computations use
 * [`IntervalArithmetic.jl`](https://github.com/JuliaIntervals/IntervalArithmetic.jl)


# Example: $SL_3(\mathbb{Z})$ has property (T)

Lets prove that $SL_3(\mathbb{Z})$ has property (T). We start with
```julia
julia> using Groups

julia> SL = MatrixGroups.SpecialLinearGroup{3}(Int8)
special linear group of 3×3 matrices over Int8
```
To define the sum of squares problem we need a group algebra:
```julia
julia> using PropertyT

julia> RSL, S, sizes = PropertyT.group_algebra(SL, gens(SL), halfradius=2, twisted=true)
[ Info: generating wl-metric ball of radius 4
  0.011135 seconds (279.78 k allocations: 9.858 MiB)
[ Info: sizes = [13, 121, 883, 5455]
[ Info: computing the *-algebra structure for G
  0.049325 seconds (158.86 k allocations: 7.218 MiB, 92.13% compilation time)
(*-algebra of special linear group of 3×3 matrices over Int8, FPGroupElement{Groups.MatrixGroups.SpecialLinearGroup{3, Int8, DataType, Alphabet{Groups.MatrixGroups.ElementaryMatrix{3, Int8}}, Vector{Groups.MatrixGroups.ElementaryMatrix{3, Int8}}}, …}[E₁₂, E₁₃, E₂₁, E₂₃, E₃₁, E₃₂, E₁₂^-1, E₁₃^-1, E₂₁^-1, E₂₃^-1, E₃₁^-1, E₃₂^-1], [13, 121, 883, 5455])
```
We supplied `halfradius=2` to be able to multiply in `RSL` algebra elements supported in the ball of radius `2` of `SL` (with the word-length metric).

## The Laplacian

We define the (symmetric) group Laplacian `Δ` using the familiar formula
```julia
julia> Δ = RSL(length(S)) - sum(RSL(s) for s in S)
12·(id) -1·E₁₂ -1·E₁₃ -1·E₂₁ -1·E₂₃ -1·E₃₁ -1·E₃₂ -1·E₁₂^-1 -1·E₁₃^-1 -1·E₂₁^-1 -1·E₂₃^-1 -1·E₃₁^-1 -1·E₃₂^-1

julia> Δ² = Δ^2
156·(id) -24·E₁₂ -24·E₁₃ -24·E₂₁ -24·E₂₃ -24·E₃₁ -24·E₃₂ -24·E₁₂^-1 -24·E₁₃^-1 -24·E₂₁^-1 -24·E₂₃^-1 -24·E₃₁^-1 -24·E₃₂^-1 +1·E₁₂^2 +2·E₁₂*E₁₃ +1·E₁₂*E₂₁ +1·E₁₂*E₂₃ +1·E₁₂*E₃₁ +2·E₁₂*E₃₂ +2·E₁₂*E₁₃^-1 +1·E₁₂*E₂₁^-1 +1·E₁₂*E₂₃^-1 +1·E₁₂*E₃₁^-1 +2·E₁₂*E₃₂^-1 +1·E₁₃^2 +1·E₁₃*E₂₁ +2·E₁₃*E₂₃ +1·E₁₃*E₃₁ +1·E₁₃*E₃₂ +2·E₁₃*E₁₂^-1 +1·E₁₃*E₂₁^-1 +2·E₁₃*E₂₃^-1 +1·E₁₃*E₃₁^-1 +1·E₁₃*E₃₂^-1 +1·E₂₁*E₁₂ +1·E₂₁*E₁₃ +1·E₂₁^2 +2·E₂₁*E₂₃ +2·E₂₁*E₃₁ +1·E₂₁*E₃₂ +1·E₂₁*E₁₂^-1 +1·E₂₁*E₁₃^-1 +2·E₂₁*E₂₃^-1 +2·E₂₁*E₃₁^-1 +1·E₂₁*E₃₂^-1 +1·E₂₃*E₁₂ +1·E₂₃^2 +1·E₂₃*E₃₁ +1·E₂₃*E₃₂ +1·E₂₃*E₁₂^-1 +2·E₂₃*E₁₃^-1 +2·E₂₃*E₂₁^-1 +1·E₂₃*E₃₁^-1 +1·E₂₃*E₃₂^-1 +1·E₃₁*E₁₂ +1·E₃₁*E₁₃ +1·E₃₁*E₂₃ +1·E₃₁^2 +2·E₃₁*E₃₂ +1·E₃₁*E₁₂^-1 +1·E₃₁*E₁₃^-1 +2·E₃₁*E₂₁^-1 +1·E₃₁*E₂₃^-1 +2·E₃₁*E₃₂^-1 +1·E₃₂*E₁₃ +1·E₃₂*E₂₁ +1·E₃₂*E₂₃ +1·E₃₂^2 +2·E₃₂*E₁₂^-1 +1·E₃₂*E₁₃^-1 +1·E₃₂*E₂₁^-1 +1·E₃₂*E₂₃^-1 +2·E₃₂*E₃₁^-1 +1·E₁₂^-1*E₂₁ +1·E₁₂^-1*E₂₃ +1·E₁₂^-1*E₃₁ +1·E₁₂^-2 +2·E₁₂^-1*E₁₃^-1 +1·E₁₂^-1*E₂₁^-1 +1·E₁₂^-1*E₂₃^-1 +1·E₁₂^-1*E₃₁^-1 +2·E₁₂^-1*E₃₂^-1 +1·E₁₃^-1*E₂₁ +1·E₁₃^-1*E₃₁ +1·E₁₃^-1*E₃₂ +1·E₁₃^-2 +1·E₁₃^-1*E₂₁^-1 +2·E₁₃^-1*E₂₃^-1 +1·E₁₃^-1*E₃₁^-1 +1·E₁₃^-1*E₃₂^-1 +1·E₂₁^-1*E₁₂ +1·E₂₁^-1*E₁₃ +1·E₂₁^-1*E₃₂ +1·E₂₁^-1*E₁₂^-1 +1·E₂₁^-1*E₁₃^-1 +1·E₂₁^-2 +2·E₂₁^-1*E₂₃^-1 +2·E₂₁^-1*E₃₁^-1 +1·E₂₁^-1*E₃₂^-1 +1·E₂₃^-1*E₁₂ +1·E₂₃^-1*E₃₁ +1·E₂₃^-1*E₃₂ +1·E₂₃^-1*E₁₂^-1 +1·E₂₃^-2 +1·E₂₃^-1*E₃₁^-1 +1·E₂₃^-1*E₃₂^-1 +1·E₃₁^-1*E₁₂ +1·E₃₁^-1*E₁₃ +1·E₃₁^-1*E₂₃ +1·E₃₁^-1*E₁₂^-1 +1·E₃₁^-1*E₁₃^-1 +1·E₃₁^-1*E₂₃^-1 +1·E₃₁^-2 +2·E₃₁^-1*E₃₂^-1 +1·E₃₂^-1*E₁₃ +1·E₃₂^-1*E₂₁ +1·E₃₂^-1*E₂₃ +1·E₃₂^-1*E₁₃^-1 +1·E₃₂^-1*E₂₁^-1 +1·E₃₂^-1*E₂₃^-1 +1·E₃₂^-2
```
## Formulating the optimization problem

As proven by N. Ozawa [here](https://arxiv.org/abs/1312.5431) (Main Theorem), property (T) for `SL` is equivalent to a sum of (hermitian) squares decomposition for $\Delta^2 - \lambda\Delta$ for some $\lambda > 0$. Let's find such decomposition using semi-definite optimization:
```julia
julia> opt_problem = PropertyT.sos_problem_primal(Δ², Δ)
A JuMP Model
Maximization problem with:
Variables: 7382
Objective function type: JuMP.VariableRef
`JuMP.AffExpr`-in-`MathOptInterface.EqualTo{Float64}`: 5455 constraints
`Vector{JuMP.VariableRef}`-in-`MathOptInterface.PositiveSemidefiniteConeTriangle`: 1 constraint
Model mode: AUTOMATIC
CachingOptimizer state: NO_OPTIMIZER
Solver name: No optimizer attached.
Names registered in the model: P, psd, λ
````
This problem tries to find maximal `λ` as long as an internal matrix `P` defines a sum of squares decomposition for `Δ² - λΔ` (you may consult the docstring of `sos_problem_primal` for more information).

## Solving the optimization problem

To solve the problem we need a solver/optimizer - a software to numerically find a solution using e.g. iterative procedures. There are two solvers predefined in `test/optimizers.jl`:
* `scs_optimizer` and
* `cosmo_optimizer`.
These are just thin wrappers around `JuMP` (or actually `MathOptInterface`) optimizers.
```julia
julia> include("test/optimizers.jl");
```

Now we have everything what we need to solve the problem!
```julia
julia> status, warmstart = PropertyT.solve(
    opt_problem,
    scs_optimizer(max_iters=5_000, accel=50, alpha=1.9),
);
------------------------------------------------------------------
               SCS v3.2.1 - Splitting Conic Solver
        (c) Brendan O'Donoghue, Stanford University, 2012
------------------------------------------------------------------
problem:  variables n: 7382, constraints m: 12836
cones:    z: primal zero / dual free vars: 5455
          s: psd vars: 7381, ssize: 1
settings: eps_abs: 1.0e-09, eps_rel: 1.0e-09, eps_infeas: 1.0e-07
          alpha: 1.90, scale: 1.00e-01, adaptive_scale: 1
          max_iters: 5000, normalize: 1, rho_x: 1.00e-06
          acceleration_lookback: 50, acceleration_interval: 10
lin-sys:  sparse-direct-amd-qdldl
          nnz(A): 57566, nnz(P): 0
------------------------------------------------------------------
 iter | pri res | dua res |   gap   |   obj   |  scale  | time (s)
------------------------------------------------------------------
     0| 1.57e+02  9.96e-01  2.61e+02 -1.69e+02  1.00e-01  6.54e-02
   250| 1.08e-02  2.03e-04  1.69e-01 -1.02e+00  1.00e-01  8.05e-01
   500| 2.94e-03  2.46e-04  5.78e-02 -5.24e-01  1.00e-01  1.55e+00
   [...]
  4500| 4.62e-06  4.05e-09  1.71e-06 -2.80e-01  6.45e-03  1.40e+01
  4750| 3.74e-06  2.92e-09  1.72e-06 -2.80e-01  6.45e-03  1.48e+01
  5000| 3.06e-06  2.03e-09  1.63e-06 -2.80e-01  6.45e-03  1.56e+01
------------------------------------------------------------------
status:  solved (inaccurate - reached max_iters)
timings: total: 1.56e+01s = setup: 6.09e-02s + solve: 1.55e+01s
         lin-sys: 1.90e+00s, cones: 1.26e+01s, accel: 1.87e-01s
------------------------------------------------------------------
objective = -0.280408 (inaccurate)
------------------------------------------------------------------

julia> status
ALMOST_OPTIMAL::TerminationStatusCode = 7
```
The solver didn't manage to solve the problem but it got quite close! (duality gap is ~`1.63e-6`). Let's try once again this time warmstarting the solver:
```julia
julia> status, warmstart = PropertyT.solve(
    opt_problem,
    scs_optimizer(max_iters=10_000, accel=50, alpha=1.9),
    warmstart,
);
------------------------------------------------------------------
               SCS v3.2.1 - Splitting Conic Solver
        (c) Brendan O'Donoghue, Stanford University, 2012
------------------------------------------------------------------
problem:  variables n: 7382, constraints m: 12836
cones:    z: primal zero / dual free vars: 5455
          s: psd vars: 7381, ssize: 1
settings: eps_abs: 1.0e-09, eps_rel: 1.0e-09, eps_infeas: 1.0e-07
          alpha: 1.90, scale: 1.00e-01, adaptive_scale: 1
          max_iters: 10000, normalize: 1, rho_x: 1.00e-06
          acceleration_lookback: 50, acceleration_interval: 10
lin-sys:  sparse-direct-amd-qdldl
          nnz(A): 57566, nnz(P): 0
------------------------------------------------------------------
 iter | pri res | dua res |   gap   |   obj   |  scale  | time (s)
------------------------------------------------------------------
     0| 3.00e-06  2.96e-08  1.37e-05 -2.80e-01  1.00e-01  4.32e-02
   250| 5.01e-07  6.75e-08  6.62e-06 -2.80e-01  1.00e-01  7.78e-01
   500| 7.79e-07  1.92e-08  1.19e-06 -2.80e-01  3.10e-02  1.58e+00
   750| 7.80e-07  7.20e-09  4.80e-07 -2.80e-01  3.10e-02  2.33e+00
  1000| 7.74e-07  5.22e-10  2.06e-07 -2.80e-01  3.10e-02  3.15e+00
  1250| 7.74e-07  3.84e-10  2.40e-07 -2.80e-01  3.10e-02  4.01e+00
  1500| 7.71e-07  1.38e-10  2.48e-07 -2.80e-01  3.10e-02  4.88e+00
  1750| 7.72e-07  2.63e-11  2.48e-07 -2.80e-01  3.10e-02  5.77e+00
  2000| 7.60e-07  1.31e-11  2.48e-07 -2.80e-01  3.10e-02  6.68e+00
  2200| 2.70e-09  1.46e-10  8.93e-10 -2.80e-01  5.37e-01  7.37e+00
------------------------------------------------------------------
status:  solved
timings: total: 7.37e+00s = setup: 3.89e-02s + solve: 7.33e+00s
         lin-sys: 8.18e-01s, cones: 6.07e+00s, accel: 6.05e-02s
------------------------------------------------------------------
objective = -0.280408
------------------------------------------------------------------

julia> status
OPTIMAL::TerminationStatusCode = 1
```
This time solver was successful in reaching the desired accuracy (`1e-9`). Lets query the solution:
```julia
julia> λ = JuMP.value(opt_problem[:λ])
0.28040750495076683

julia> P = JuMP.value.(opt_problem[:P]); size(P)
(121, 121)

julia> Q = real.(sqrt(P));

julia> maximum(abs, Q'*Q - P)
7.951418690144152e-11
```

## Certifying the result

Thus we obtained a matrix `Q` which defines elements `ξᵢ ∈ ℝSL` (coefficents read by columns of `Q`) whose sum of squares is close to `Δ²-λΔ` in `ℓ₁`-norm. Let's check it out.

```julia
julia> sos = PropertyT.compute_sos(RSL, Q, augmented=true);

julia> using LinearAlgebra

julia> norm(Δ²-λ*Δ - sos, 1)
2.948008083982383e-7
```
We'd like to conclude from this that since the norm of the residual is much larger than `λ` we obtain (by `ℓ₁`-convexity of sum of squares cone in `ℝSL`) a proof of the existence of an **exact** sum of squares decomposition of `Δ² - λ₀Δ` for some `λ₀` not far from the numerical `λ` above. To be able to do so we'd need to provide a certified bound on the magnitude of the norm. Here's how to do it in an automated fashion:
```julia
julia> _, λ_cert = PropertyT.certify_solution(Δ², Δ, λ, Q, halfradius=2, augmented=true)
  0.070032 seconds (4.11 k allocations: 400.047 KiB)
┌ Info: Checking in Float64 arithmetic with
└   λ = 0.28040750495076683
┌ Info: Numerical metrics of the obtained SOS:
│ ɛ(elt - λu - ∑ξᵢ*ξᵢ) ≈ -3.3119650146808302e-12
│ ‖elt - λu - ∑ξᵢ*ξᵢ‖₁ ≈ 2.948008083982383e-7
└  λ ≈ 0.2804063257475332
  5.393452 seconds (15.30 M allocations: 581.181 MiB, 3.08% gc time, 83.70% compilation time)
┌ Info: Checking in IntervalArithmetic.Interval{Float64} arithmetic with
└   λ = 0.28040750495076683
┌ Info: Numerical metrics of the obtained SOS:
│ ɛ(elt - λu - ∑ξᵢ*ξᵢ) ∈ [-1.92655e-10, 2.00372e-10]
│ ‖elt - λu - ∑ξᵢ*ξᵢ‖₁ ∈ [2.94597e-07, 2.94991e-07]
└  λ ∈ [0.280406, 0.280407]
(true, 0.28040632499038354)
```
The true returned means that the result is certified to be correct and the value `0.28040632499038354` is the certified lower bound on the spectral gap of `Δ`.

### Lower bound on the Kazhdan constant
Together with estimate for the Kazhdan constant of $(G,S)$:
$$\sqrt{\frac{2\lambda(G,S)}{\vert S\vert}} \leqslant \kappa(G,S)$$
we obtain
```julia
julia> sqrt(2λ/length(S))
0.21618183124041931
```
hence $0.2161... \leqslant \kappa(SL_3(\mathbb{Z}), S_3)$, i.e. $SL_3(\mathbb{Z})$ has property (T).

