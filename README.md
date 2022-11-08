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
