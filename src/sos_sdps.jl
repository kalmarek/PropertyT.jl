"""
    sos_problem_dual(X, [u = zero(X); upper_bound=Inf])
Formulate the dual to the sum of squares decomposition problem for `X - λ·u`.

See also [sos_problem_primal](@ref).
"""
function sos_problem_dual(
    elt::StarAlgebras.AlgebraElement,
    order_unit::StarAlgebras.AlgebraElement = zero(elt);
    lower_bound = -Inf,
)
    @assert parent(elt) == parent(order_unit)
    algebra = parent(elt)
    moment_matrix = let m = algebra.mstructure
        [m[-i, j] for i in axes(m, 1) for j in axes(m, 2)]
    end

    # 1 variable for every primal constraint
    # 1 dual variable for every element of basis
    # Symmetrized:
    # 1 dual variable for every orbit of G acting on basis
    model = Model()
    @variable model y[1:length(basis(algebra))]
    @constraint model λ_dual dot(order_unit, y) == 1
    @constraint(model, psd, y[moment_matrix] in PSDCone())

    if !isinf(lower_bound)
        throw("Not Implemented yet")
        @variable model λ_ub_dual
        @objective model Min dot(elt, y) + lower_bound * λ_ub_dual
    else
        @objective model Min dot(elt, y)
    end

    return model
end

"""
    sos_problem_primal(X, [u = zero(X); upper_bound=Inf])
Formulate sum of squares decomposition problem for `X - λ·u`.

Given element `X` of a star-algebra `A` and an order unit `u` of `Σ²A` the sum
of squares cone in `A`, fomulate sum of squares decomposition problem:

```
max:        λ
subject to: X - λ·u ∈ Σ²A
```

If `upper_bound` is given a finite value, additionally `λ ≤ upper_bound` will
be added to the model. This may improve the accuracy of the solution if
`upper_bound` is less than the optimal `λ`.

The default `u = zero(X)` formulates a simple feasibility problem.
"""
function sos_problem_primal(
    elt::StarAlgebras.AlgebraElement,
    order_unit::StarAlgebras.AlgebraElement = zero(elt);
    upper_bound = Inf,
    augmented::Bool = iszero(StarAlgebras.aug(elt)) &&
                      iszero(StarAlgebras.aug(order_unit)),
)
    @assert parent(elt) === parent(order_unit)

    N = LinearAlgebra.checksquare(parent(elt).mstructure)
    model = JuMP.Model()
    P = JuMP.@variable(model, P[1:N, 1:N], Symmetric)
    JuMP.@constraint(model, psd, P in PSDCone())

    if iszero(order_unit) && isfinite(upper_bound)
        @warn "Setting `upper_bound` together with zero `order_unit` has no effect"
    end

    A = constraints(parent(elt); augmented = augmented)

    if !iszero(order_unit)
        λ = JuMP.@variable(model, λ)
        if isfinite(upper_bound)
            JuMP.@constraint model λ <= upper_bound
        end
        JuMP.@objective(model, Max, λ)

        for b in basis(parent(elt))
            JuMP.@constraint(model, elt(b) - λ * order_unit(b) == dot(A[b], P))
        end
    else
        for b in basis(parent(elt))
            JuMP.@constraint(model, elt(b) == dot(A[b], P))
        end
    end

    return model
end

function invariant_constraint!(
    result::AbstractMatrix,
    basis::StarAlgebras.AbstractBasis,
    cnstrs::AbstractDict{K,<:ConstraintMatrix},
    invariant_vec::SparseVector,
) where {K}
    result .= zero(eltype(result))
    for i in SparseArrays.nonzeroinds(invariant_vec)
        g = basis[i]
        A = cnstrs[g]
        for (idx, v) in nzpairs(A)
            result[idx] += invariant_vec[i] * v
        end
    end
    return result
end

function isorth_projection(ds::SymbolicWedderburn.DirectSummand)
    U = SymbolicWedderburn.image_basis(ds)
    return isapprox(U * U', I)
end

function sos_problem_primal(
    elt::StarAlgebras.AlgebraElement,
    wedderburn::WedderburnDecomposition;
    kwargs...,
)
    return sos_problem_primal(elt, zero(elt), wedderburn; kwargs...)
end

function __fast_recursive_dot!(
    res::JuMP.AffExpr,
    Ps::AbstractArray{<:AbstractMatrix{<:JuMP.VariableRef}},
    Ms::AbstractArray{<:AbstractSparseMatrix},
    weights,
)
    @assert length(Ps) == length(Ms)

    for (w, A, P) in zip(weights, Ms, Ps)
        iszero(Ms) && continue
        rows = rowvals(A)
        vals = nonzeros(A)
        for cidx in axes(A, 2)
            for i in nzrange(A, cidx)
                ridx = rows[i]
                val = vals[i]
                JuMP.add_to_expression!(res, P[ridx, cidx], w * val)
            end
        end
    end
    return res
end

function _dot(
    Ps::AbstractArray{<:AbstractMatrix{<:JuMP.VariableRef}},
    Ms::AbstractArray{<:AbstractMatrix{T}},
    weights = Iterators.repeated(one(T), length(Ms)),
) where {T}
    return __fast_recursive_dot!(JuMP.AffExpr(), Ps, Ms, weights)
end

import ProgressMeter
__show_itrs(n, total) = () -> [(Symbol("constraint"), "$n/$total")]

function sos_problem_primal(
    elt::StarAlgebras.AlgebraElement,
    orderunit::StarAlgebras.AlgebraElement,
    wedderburn::WedderburnDecomposition;
    upper_bound = Inf,
    augmented = iszero(StarAlgebras.aug(elt)) &&
        iszero(StarAlgebras.aug(orderunit)),
    check_orthogonality = true,
    show_progress = false,
)
    @assert parent(elt) === parent(orderunit)
    if check_orthogonality
        if any(!isorth_projection, direct_summands(wedderburn))
            error(
                "Wedderburn decomposition contains a non-orthogonal projection",
            )
        end
    end

    feasibility_problem = iszero(orderunit)

    model = JuMP.Model()
    if !feasibility_problem # add λ or not?
        λ = JuMP.@variable(model, λ)
        JuMP.@objective(model, Max, λ)

        if isfinite(upper_bound)
            JuMP.@constraint(model, λ <= upper_bound)
            if feasibility_problem
                @warn "setting `upper_bound` with zero `orderunit` has no effect"
            end
        end
    end

    P = map(direct_summands(wedderburn)) do ds
        dim = size(ds, 1)
        P = JuMP.@variable(model, [1:dim, 1:dim], Symmetric)
        JuMP.@constraint(model, P in PSDCone())
        return P
    end

    begin # preallocating
        T = eltype(wedderburn)
        Ms = [spzeros.(T, size(p)...) for p in P]
        M_orb = zeros(T, size(parent(elt).mstructure)...)
    end

    X = convert(Vector{T}, StarAlgebras.coeffs(elt))
    U = convert(Vector{T}, StarAlgebras.coeffs(orderunit))

    # defining constraints based on the multiplicative structure
    cnstrs = constraints(parent(elt); augmented = augmented)

    prog = ProgressMeter.Progress(
        length(invariant_vectors(wedderburn));
        dt = 1,
        desc = "Adding constraints: ",
        enabled = show_progress,
        barlen = 60,
        showspeed = true,
    )

    for (i, iv) in enumerate(invariant_vectors(wedderburn))
        ProgressMeter.next!(prog; showvalues = __show_itrs(i, prog.n))

        x = dot(X, iv)
        u = dot(U, iv)

        M_orb = invariant_constraint!(M_orb, basis(parent(elt)), cnstrs, iv)

        Ms = SymbolicWedderburn.diagonalize!(
            Ms,
            M_orb,
            wedderburn;
            trace_preserving = true,
        )
        # SparseArrays.droptol!.(Ms, 10 * eps(T) * max(size(M_orb)...))

        # @info [nnz(m) / length(m) for m in Ms]

        if feasibility_problem
            JuMP.@constraint(model, x == _dot(P, Ms))
        else
            JuMP.@constraint(model, x - λ * u == _dot(P, Ms))
        end
    end
    ProgressMeter.finish!(prog)
    return model, P
end
