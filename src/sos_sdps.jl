"""
    sos_problem_dual(X, [u = zero(X); upper_bound=Inf])
Formulate the dual to the sum of squares decomposition problem for `X - λ·u`.

See also [sos_problem_primal](@ref).
"""
function sos_problem_dual(
    elt::StarAlgebras.AlgebraElement,
    order_unit::StarAlgebras.AlgebraElement=zero(elt);
    lower_bound=-Inf
)
    @assert parent(elt) == parent(order_unit)
    algebra = parent(elt)
    mstructure = if StarAlgebras._istwisted(algebra.mstructure)
        algebra.mstructure
    else
        StarAlgebras.MTable{true}(basis(algebra), table_size=size(algebra.mstructure))
    end

    # 1 variable for every primal constraint
    # 1 dual variable for every element of basis
    # Symmetrized:
    # 1 dual variable for every orbit of G acting on basis
    model = Model()
    @variable model y[1:length(basis(algebra))]
    @constraint model λ_dual dot(order_unit, y) == 1
    @constraint(model, psd, y[mstructure] in PSDCone())

    if !isinf(lower_bound)
        throw("Not Implemented yet")
        @variable model λ_ub_dual
        @objective model Min dot(elt, y) + lower_bound * λ_ub_dual
    else
        @objective model Min dot(elt, y)
    end

    return model
end

function constraints(
    basis::StarAlgebras.AbstractBasis,
    mstr::AbstractMatrix{<:Integer};
    augmented::Bool=false,
    table_size=size(mstr)
)
    cnstrs = [signed(eltype(mstr))[] for _ in basis]
    LI = LinearIndices(table_size)

    for ci in CartesianIndices(table_size)
        k = LI[ci]
        a_star_b = basis[mstr[k]]
        push!(cnstrs[basis[a_star_b]], k)
        if augmented
            # (1-a_star)(1-b) = 1 - a_star - b + a_star_b

            i, j = Tuple(ci)
            a, b = basis[i], basis[j]

            push!(cnstrs[basis[one(a)]], k)
            push!(cnstrs[basis[StarAlgebras.star(a)]], -k)
            push!(cnstrs[basis[b]], -k)
        end
    end

    return Dict(
        basis[i] => ConstraintMatrix(c, table_size..., 1) for (i, c) in pairs(cnstrs)
    )
end

function constraints(A::StarAlgebras.StarAlgebra; augmented::Bool, twisted::Bool)
    mstructure = if StarAlgebras._istwisted(A.mstructure) == twisted
        A.mstructure
    else
        StarAlgebras.MTable{twisted}(basis(A), table_size=size(A.mstructure))
    end
    return constraints(basis(A), mstructure, augmented=augmented)
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
    order_unit::StarAlgebras.AlgebraElement=zero(elt);
    upper_bound=Inf,
    augmented::Bool=iszero(StarAlgebras.aug(elt)) && iszero(StarAlgebras.aug(order_unit))
)
    @assert parent(elt) === parent(order_unit)

    N = LinearAlgebra.checksquare(parent(elt).mstructure)
    model = JuMP.Model()
    P = JuMP.@variable(model, P[1:N, 1:N], Symmetric)
    JuMP.@constraint(model, psd, P in PSDCone())

    if iszero(order_unit) && isfinite(upper_bound)
        @warn "Setting `upper_bound` together with zero `order_unit` has no effect"
    end

    A = constraints(parent(elt), augmented=augmented, twisted=true)

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
    cnstrs::AbstractDict{K,CM},
    invariant_vec::SparseVector,
) where {K,CM<:ConstraintMatrix}
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

sos_problem_primal(
    elt::StarAlgebras.AlgebraElement,
    wedderburn::WedderburnDecomposition;
    kwargs...
) = sos_problem_primal(elt, zero(elt), wedderburn; kwargs...)

function __fast_recursive_dot!(
    res::JuMP.AffExpr,
    Ps::AbstractArray{<:AbstractMatrix{<:JuMP.VariableRef}},
    Ms::AbstractArray{<:AbstractSparseMatrix};
)
    @assert length(Ps) == length(Ms)

    for (A, P) in zip(Ms, Ps)
        iszero(Ms) && continue
        rows = rowvals(A)
        vals = nonzeros(A)
        for cidx in axes(A, 2)
            for i in nzrange(A, cidx)
                ridx = rows[i]
                val = vals[i]
                JuMP.add_to_expression!(res, P[ridx, cidx], val)
            end
        end
    end
    return res
end

import ProgressMeter
__show_itrs(n, total) = () -> [(Symbol("constraint"), "$n/$total")]

function sos_problem_primal(
    elt::StarAlgebras.AlgebraElement,
    orderunit::StarAlgebras.AlgebraElement,
    wedderburn::WedderburnDecomposition;
    upper_bound=Inf,
    augmented=iszero(StarAlgebras.aug(elt)) && iszero(StarAlgebras.aug(orderunit)),
    check_orthogonality=true,
    show_progress=false
)

    @assert parent(elt) === parent(orderunit)
    if check_orthogonality
        if any(!isorth_projection, direct_summands(wedderburn))
            error("Wedderburn decomposition contains a non-orthogonal projection")
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
        P
    end

    begin # preallocating
        T = eltype(wedderburn)
        Ms = spzeros.(T, size.(P))
        M_orb = zeros(T, size(parent(elt).mstructure)...)
    end

    X = convert(Vector{T}, StarAlgebras.coeffs(elt))
    U = convert(Vector{T}, StarAlgebras.coeffs(orderunit))

    # defining constraints based on the multiplicative structure
    cnstrs = constraints(parent(elt), augmented=augmented, twisted=true)

    prog = ProgressMeter.Progress(
        length(invariant_vectors(wedderburn)),
        dt=1,
        desc="Adding constraints... ",
        enabled=show_progress,
        barlen=60,
        showspeed=true
    )

    for (i, iv) in enumerate(invariant_vectors(wedderburn))
        ProgressMeter.next!(prog, showvalues=__show_itrs(i, prog.n))

        x = dot(X, iv)
        u = dot(U, iv)

        M_orb = invariant_constraint!(M_orb, basis(parent(elt)), cnstrs, iv)
        Ms = SymbolicWedderburn.diagonalize!(Ms, M_orb, wedderburn)
        SparseArrays.droptol!.(Ms, 10 * eps(T) * max(size(M_orb)...))

        # @info [nnz(m) / length(m) for m in Ms]

        if feasibility_problem
            JuMP.@constraint(
                model,
                x == __fast_recursive_dot!(JuMP.AffExpr(), P, Ms)
            )
        else
            JuMP.@constraint(
                model,
                x - λ * u == __fast_recursive_dot!(JuMP.AffExpr(), P, Ms)
            )
        end
    end
    ProgressMeter.finish!(prog)
    return model, P
end

function reconstruct(Ps, wd::WedderburnDecomposition)
    N = size(first(direct_summands(wd)), 2)
    P = zeros(eltype(wd), N, N)
    return reconstruct!(P, Ps, wd)
end

function group_of(wd::WedderburnDecomposition)
    # this is veeeery hacky... ;)
    return parent(first(keys(wd.hom.cache)))
end

# TODO: move to SymbolicWedderburn
SymbolicWedderburn.action(wd::WedderburnDecomposition) =
    SymbolicWedderburn.action(wd.hom)

function reconstruct!(
    res::AbstractMatrix,
    Ps,
    wedderburn::WedderburnDecomposition,
)
    G = group_of(wedderburn)

    act = SymbolicWedderburn.action(wedderburn)

    @assert act isa SymbolicWedderburn.ByPermutations

    for (π, ds) in pairs(direct_summands(wedderburn))
        Uπ = SymbolicWedderburn.image_basis(ds)

        # LinearAlgebra.mul!(tmp, Uπ', P[π])
        # LinearAlgebra.mul!(tmp2, tmp, Uπ)
        tmp2 = Uπ' * Ps[π] * Uπ
        if eltype(res) <: AbstractFloat
            SymbolicWedderburn.zerotol!(tmp2, atol=1e-12)
        end
        tmp2 .*= SymbolicWedderburn.degree(ds)

        @assert size(tmp2) == size(res)

        for g in G
            p = SymbolicWedderburn.induce(wedderburn.hom, g)
            for c in axes(res, 2)
                for r in axes(res, 1)
                    res[r, c] += tmp2[r^p, c^p]
                end
            end
        end
    end
    res ./= Groups.order(Int, G)

    return res
end

