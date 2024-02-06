"""
    sos_problem_dual(X, [u = zero(X); upper_bound=Inf])
Formulate the dual to the sum of squares decomposition problem for `X - λ·u`.

See also [sos_problem_primal](@ref).
"""
function sos_problem_dual(
    elt::StarAlgebras.AlgebraElement,
    order_unit::StarAlgebras.AlgebraElement = zero(elt);
    lower_bound = -Inf,
    supp::AbstractVector{<:Integer} = axes(parent(elt).mstructure, 1),
)
    @assert parent(elt) == parent(order_unit)
    A = parent(elt)

    # 1 variable for every primal constraint
    # 1 dual variable for every element of basis
    # Symmetrized:
    # 1 dual variable for every orbit of G acting on basis
    model = Model()
    JuMP.@variable(model, y[1:length(basis(A))])
    JuMP.@constraint(model, λ_dual, dot(order_unit, y) == 1)
    let mstr = A.mstructure
        moment_matrix = [mstr[-i, j] for i in supp, j in supp]
        JuMP.@constraint(model, psd, y[moment_matrix] in PSDCone())
    end

    if !isinf(lower_bound)
        throw("Not Implemented yet")
        JuMP.@variable(model, λ_ub_dual)
        JuMP.@objective(model, Min, dot(elt, y) + lower_bound * λ_ub_dual)
    else
        JuMP.@objective(model, Min, dot(elt, y))
    end

    return model
end

function geometric_constraints!(
    model::JuMP.Model,
    elt::StarAlgebras.AlgebraElement,
)
    A = parent(elt)
    G = parent(A)
    mstr = A.mstructure
    b = basis(A)
    y = model[:y]
    for g in gens(G)
        for h in gens(G)
            gh = mstr[b[g], b[h]]
            if elt[gh] > 0
                for γ in axes(mstr, 1)
                    γgh = mstr[γ, gh]
                    γg = mstr[γ, b[g]]
                    γh = mstr[γ, b[h]]
                    JuMP.@constraint model y[γgh] + y[γ] == y[γg] + y[γh]
                end
            end
        end
    end
    return model
end

function decompose(
    elt::StarAlgebras.AlgebraElement,
    wd::WedderburnDecomposition,
)
    v = StarAlgebras.coeffs(elt)
    cfs, error = decompose(v, invariant_vectors(wd))
    _eps = length(v) * eps(typeof(error))
    error < _eps || @warn "elt does not seem to be invariant" error
    return cfs
end

function decompose(v::AbstractVector, invariant_vecs)
    # TODO: eltype for current, res ?
    current = similar(v, Float64)
    current .= 0.0
    res = SparseArrays.spzeros(length(invariant_vecs))

    _eps = length(current) * eps(eltype(res))
    diff = zero(current)

    for (i, iv) in enumerate(invariant_vecs)
        cf = dot(v, iv) / dot(iv, iv)
        if !iszero(cf)
            res[i] = cf
            current .+= cf .* iv
            diff .= current .- v
            if norm(diff) < _eps
                break
            end
        end
    end
    return res, norm(current - v)
end

function _dot(elt::StarAlgebras.AlgebraElement, Y, wd::WedderburnDecomposition)
    inv_vecs = invariant_vectors(wd)
    v = StarAlgebras.coeffs(elt)
    res, error = _dot(v, Y, inv_vecs)
    _eps = length(v) * eps(typeof(error))
    error < _eps || @warn "elt does not seem to be invariant" error
    return res
end

function _dot(v::AbstractVector, Y::AbstractVector{<:JuMP.AbstractVariableRef})
    @assert length(inv_vecs) == length(Y)
    @assert length(v) == length(first(inv_vecs))
    res = JuMP.AffExpr()

    cfs, error = decompose(v, inv_vecs)

    for i in SparseArrays.nonzeroinds(cfs)
        (c, y) = cfs[i], Y[i]
        JuMP.add_to_expression!(res, c, y)
    end
    return res, error
end

function sos_problem_dual(
    elt::StarAlgebras.AlgebraElement,
    order_unit::StarAlgebras.AlgebraElement,
    wd::WedderburnDecomposition,
    lower_bound = -Inf,
    supp::AbstractVector{<:Integer} = axes(parent(elt).mstructure, 1),
)
    @assert parent(elt) == parent(order_unit)
    inv_vecs = invariant_vectors(wd)

    model = Model()
    # 1 dual variable per orbit of G acting on basis
    JuMP.@variable(model, y_orb[1:length(inv_vecs)])

    # the value of y on order_unit is 1 (y is normalized)
    let unit_orbit_cfs = decompose(order_unit, wd)
        JuMP.@constraint(model, λ_dual, dot(unit_orbit_cfs, y_orb) == 1)
    end

    # here we reconstruct the original y;
    # TODO: this is **BAD**; do something about it
    # y = sum(y .* iv for (y, iv) in zip(y_orb, invariant_vectors(wd)))
    y = let y = [JuMP.AffExpr() for _ in 1:length(first(invariant_vectors(wd)))]
        for (y_o, iv) in zip(y_orb, invariant_vectors(wd))
            for i in SparseArrays.nonzeroinds(iv)
                JuMP.add_to_expression!(y[i], nnz(iv) * iv[i], y_o)
            end
        end
        y
    end
    Ps = let mstr = parent(elt).mstructure, y = y
        moment_matrix = [mstr[-i, j] for i in supp, j in supp]
        # JuMP.@constraint(model, psd, y[moment_matrix] in PSDCone())
        Ps = SymbolicWedderburn.diagonalize(y[moment_matrix], wd)
        for P in Ps
            JuMP.@constraint(model, P in PSDCone())
        end
        Ps
    end

    if !isinf(lower_bound)
        throw("Not Implemented yet")
        JuMP.@variable(model, λ_ub_dual)
        # JuMP.@objective(model, Min, _dot(elt, y_orb, wd) + lower_bound * λ_ub_dual)
    else
        let elt_orbit_cfs = decompose(elt, wd)
            JuMP.@objective(model, Min, dot(elt_orbit_cfs, y_orb, wd))
        end
    end

    return model, Ps
end

function __find_firstnz(i, inv_vecs)
    for (idx, iv) in enumerate(inv_vecs)
        iv[i] ≠ 0 && return idx
    end
    return nothing
end

function geometric_constraints!(
    model::JuMP.Model,
    elt::StarAlgebras.AlgebraElement,
    wd::WedderburnDecomposition,
)
    A = parent(elt)
    G = parent(A)
    mstr = A.mstructure
    b = basis(A)
    y = model[:y_orb]
    # cfs = PropertyT.decompose(elt, wd)
    inv_vecs = invariant_vectors(wd)
    for g in gens(G)
        for h in gens(G)
            gh = mstr[b[g], b[h]]
            if elt[gh] > 0
                for (γ, iv) in enumerate(inv_vecs)
                    γ_basis_idx = first(SparseArrays.nonzeroinds(iv))
                    γ_basis_idx > size(mstr, 1) && break

                    γgh = __find_firstnz(mstr[γ_basis_idx, gh], inv_vecs)
                    γg = __find_firstnz(mstr[γ_basis_idx, b[g]], inv_vecs)
                    γh = __find_firstnz(mstr[γ_basis_idx, b[h]], inv_vecs)

                    JuMP.@constraint model y[γgh] + y[γ] == y[γg] + y[γh]
                end
            end
        end
    end
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
    support = 1:size(parent(elt).mstructure, 1),
)
    @assert parent(elt) === parent(order_unit)

    N = length(support)
    model = JuMP.Model()
    P = JuMP.@variable(model, P[1:N, 1:N], Symmetric)
    JuMP.@constraint(model, psd, P in PSDCone())

    if iszero(order_unit) && isfinite(upper_bound)
        @warn "Setting `upper_bound` together with zero `order_unit` has no effect"
    end

    A = constraints(parent(elt), support; augmented = augmented)

    if !iszero(order_unit)
        λ = JuMP.@variable(model, λ)
        if !isfinite(upper_bound)
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
    @inbounds for i in SparseArrays.nonzeroinds(invariant_vec)
        g = basis[i]
        A = cnstrs[g]
        for (idx, v) in nzpairs(A)
            result[idx] += invariant_vec[i] * v
        end
    end
    return result
end

function invariant_constraint(basis, cnstrs, invariant_vec)
    I = UInt32[]
    J = UInt32[]
    V = Float64[]
    _M = first(values(cnstrs))
    CI = CartesianIndices(_M)
    @inbounds for i in SparseArrays.nonzeroinds(invariant_vec)
        g = basis[i]
        A = cnstrs[g]
        for (idx, v) in nzpairs(A)
            ci = CI[idx]
            push!(I, ci[1])
            push!(J, ci[2])
            push!(V, invariant_vec[i] * v)
        end
    end
    return sparse(I, J, V, size(_M)...)
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

    id_one = let b = basis(parent(elt)), iv = invariant_vectors(wedderburn)
        id_v = sparsevec([b[one(first(b))]], [1 // 1], length(first(iv)))
        findfirst(==(id_v), iv)
    end

    prog = ProgressMeter.Progress(
        length(invariant_vectors(wedderburn));
        dt = 1,
        desc = "Adding constraints: ",
        enabled = show_progress,
        barlen = 60,
        showspeed = true,
    )

    feasibility_problem = iszero(orderunit)

    # problem creation starts here
    model = JuMP.Model()
    if !feasibility_problem # add λ or not?
        λ = JuMP.@variable(model, λ)
        JuMP.@objective(model, Max, λ)
    end
    if isfinite(upper_bound)
        if feasibility_problem
            @warn "setting `upper_bound` with zero `orderunit` has no effect"
        else
            JuMP.@constraint(model, ub, λ <= upper_bound)
        end
    end

    # semidefinite constraints as described by wedderburn
    Ps = map(direct_summands(wedderburn)) do ds
        dim = size(ds, 1)
        P = JuMP.@variable(model, [1:dim, 1:dim], Symmetric)
        JuMP.@constraint(model, P in PSDCone())
        return P
    end

    begin # Ms are preallocated for the constraints loop
        T = eltype(wedderburn)
        Ms = [spzeros.(T, size(p)...) for p in Ps]
        _eps = 10 * eps(T) * max(size(parent(elt).mstructure)...)
    end

    X = StarAlgebras.coeffs(elt)
    U = StarAlgebras.coeffs(orderunit)

    # defining constraints based on the multiplicative structure
    cnstrs = constraints(parent(elt); augmented = augmented)

    # adding linear constraints: one per orbit
    for (i, iv) in enumerate(invariant_vectors(wedderburn))
        ProgressMeter.next!(prog; showvalues = __show_itrs(i, prog.n))
        augmented && i == id_one && continue
        # i == 500 && break

        x = dot(X, iv)
        u = dot(U, iv)

        spM_orb = invariant_constraint(basis(parent(elt)), cnstrs, iv)

        Ms = SymbolicWedderburn.diagonalize!(
            Ms,
            spM_orb,
            wedderburn;
            trace_preserving = true,
        )
        for M in Ms
            SparseArrays.droptol!(M, _eps)
        end
        if feasibility_problem
            JuMP.@constraint(model, x == _dot(Ps, Ms))
        else
            JuMP.@constraint(model, x - λ * u == _dot(Ps, Ms))
        end
    end
    ProgressMeter.finish!(prog)
    return model, Ps
end
