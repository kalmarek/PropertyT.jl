# Property(T)

[![Build Status](https://travis-ci.org/kalmarek/PropertyT.jl.svg?branch=master)](https://travis-ci.org/kalmarek/PropertyT.jl)
[![codecov](https://codecov.io/gh/kalmarek/PropertyT.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kalmarek/PropertyT.jl)

This package is concerned with sum of squares decompositions in group rings of finitely presented groups.
Please have a look into [test](https://github.com/kalmarek/GroupRings.jl/blob/master/test/runtests.jl) directory to see how to use this package. For an example applications have a look at our papers:
[1703.09680](https://arxiv.org/abs/1703.09680), [1712.07167](https://arxiv.org/abs/1712.07167) and [1812.03456](https://arxiv.org/abs/1812.03456).

The package depends on
 * [AbstractAlgebra](https://github.com/Nemocas/AbstractAlgebra.jl),
 * [Groups](https://github.com/kalmarek/Groups.jl)
 * [GroupRings](https://github.com/kalmarek/GroupRings.jl)
 * [JuMP](https://github.com/JuliaOpt/JuMP.jl)
 * [scs](https://github.com/JuliaOpt/SCS.jl) [solver](https://github.com/cvxgrp/scs)
