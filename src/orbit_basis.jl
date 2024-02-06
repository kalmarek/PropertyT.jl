function minimal_in_orbit(f, x, group, act::SymbolicWedderburn.Action)
    stab_size = UInt32(0)
    min, min_elt = f(x), x
    for g in group
        y = action(act, g, x)
        if y == x
            stab_size += 1
            continue
        else
            fy = f(y)
            if fy < min
                min, min_elt = fy, y
            end
        end
    end

    return min_elt, stab_size
end

function saturate!(
    f,
    gens,
    elements,
    stab_sizes,
    hashes = Set(elements);
    start_at = firstindex(elements),
)
    g_shifted = similar(gens)
    s_sizes = similar(gens, UInt32)
    already_seen = similar(gens, Bool)
    range = start_at:lastindex(elements)
    ProgressMeter.@showprogress 2.0 "Computing Orbit representatives: $(length(range))" for idx in
                                                                                            range
        g = elements[idx]

        Threads.@threads for i in eachindex(gens)
            h, s_size = f(g * gens[i])
            g_shifted[i] = h
            s_sizes[i] = s_size
            already_seen[i] = h in hashes
        end
        # t = g_shifted[.!seen]
        # t = unique!(t)
        # append!(old, t)
        # union!(hashes, t)
        for (seen, g, ssize) in zip(already_seen, g_shifted, s_sizes)
            if !seen
                if !(g in hashes)
                    push!(elements, g)
                    push!(stab_sizes, ssize)
                    push!(hashes, g)
                end
            end
        end
    end
    return elements, stab_sizes, hashes
end

function orbit_reps(
    seed_elts,
    Σ::Groups.Group,
    act::SymbolicWedderburn.Action;
    radius::Integer,
    minimize = hash,
)
    @assert !isempty(seed_elts)
    @assert radius ≥ 0

    if Groups.order(Int, Σ) < 2^15
        f = x -> minimal_in_orbit(minimize, x, collect(Σ), act)
    else
        f = x -> minimal_in_orbit(minimize, x, Σ, act)
    end

    orbits = [one(first(seed_elts))]
    stab_sizes = [order(UInt32, Σ)]
    hashes = Set(orbits)
    sizes = ones(Int, radius + 1)

    start_at = sizes[1]
    for r in 1:radius
        orbits, hashes = saturate!(
            f,
            seed_elts,
            orbits,
            stab_sizes,
            hashes;
            start_at = start_at,
        )
        sizes[r+1] = length(orbits)
        start_at = sizes[r]
    end

    return (orbits, stab_sizes), sizes[begin+1:end]
end

###
# move to StarAlgebras

import SparseArrays
function SparseArrays.indtype(b::StarAlgebras.AbstractBasis)
    return SparseArrays.indtype(typeof(b))
end
function SparseArrays.indtype(
    ::Type{<:StarAlgebras.AbstractBasis{T,I}},
) where {T,I}
    return I
end

###

struct OrbitBasis{T,I,G,A,F} <: StarAlgebras.AbstractBasis{T,I}
    orbit_reps::StarAlgebras.Basis{T,I}
    stab_size::Vector{UInt32}
    group::G
    action::A
    __minimize_function::F

    function OrbitBasis(
        orbit_reps::StarAlgebras.AbstractBasis{T,I},
        grp,
        act::SymbolicWedderburn.Action,
        minimize,
        stab_size::AbstractVector{<:Integer},
    ) where {T,I}
        return new{T,I,typeof(grp),typeof(act),typeof(minimize)}(
            orbits_reps,
            grp,
            act,
            stab_size,
            minimize,
        )
    end
end

__group_of(ob::OrbitBasis) = ob.group
SymbolicWedderburn.action(ob::OrbitBasis) = ob.action

Base.size(ob::OrbitBasis) = size(ob.orbit_reps)
Base.getindex(ob::OrbitBasis, i::Integer) = ob.orbit_reps[i]
function Base.getindex(ob::OrbitBasis{T}, x::T) where {T}
    if x in ob.orbit_reps
        return ob.orbit_reps[x]
    else
        for σ in __group_of(ob)
            y = action(action(ob), σ, x)
            y in ob.orbit_reps && return ob.orbit_reps[y]
        end
        argmin(ob.__minimize_function)
        throw(KeyError(minimal_in_orbit(hash, x, __group_of(ob), action(ob))))
        # y = minimal_in_orbit(hash, x, __group_of(ob), action(ob))
        # y in ob.basis ? return ob.basis[y] : throw("NotInBasis")
    end
end

function _mtable(
    basis::StarAlgebras.AbstractBasis{T},
    orbits::OrbitBasis{T},
    table_size,
) where {T}
    M = zeros(SparseArrays.indtype(orbits), table_size)
    starof = zeros(eltype(M), size(M, 1))

    # it is safe to do Threads.@threads here, since
    # in case of race condition (two threads overwriting a single entry in M)
    # * M[idx, jdx] is 0 at the beginning,
    # * M[idx, jdx] will be written by two threads
    # thread, we will be computing and overwriting with the same value;
    Threads.@threads for jdx in axes(M, 2)
        b = basis[jdx]
        starof[jdx] = basis[StarAlgebras.star(b)]
        for idx in axes(M, 1)
            !iszero(M[idx, jdx]) && continue
            a = basis[idx]
            k = orbits[a*b]
            for g in orbits.group
                aᵍ = action(action(orbits), g, a)
                bᵍ = action(action(orbits), g, b)
                M[basis[aᵍ], basis[bᵍ]] = k
            end
        end
    end

    @assert all(!iszero, starof)
    @assert all(!iszero, M)
    return StarAlgebras.MTable(M, starof)
end

function constraints(
    basis_half::StarAlgebras.AbstractBasis,
    orbits::OrbitBasis;
    augmented::Bool,
    basis_support,
)
    l = length(basis_support)
    mstr = _mtable(basis_half, orbits, (l, l))
    id = orbits[one(first(basis_half))]
    cnstrs = _constraints(mstr; augmented = augmented, id = id)

    ordG = length(__group_of(orbits))

    return Dict(
        orbits[i] =>
            ConstraintMatrix(c, size(mstr)..., orbits.stab_size[i] / ordG) for
        (i, c) in pairs(cnstrs) if !isempty(c)
    )
end

## move to Tests
#=
mt = let R = 2, sizes = sizes
    mt = @time _mtable(basis(RG), obasis, (sizes[R], sizes[R]))
    # k = findlast(o -> basis(RG)[o] ≤ sizes[R], orbits)
    # @assert k == osizes[2R]
    @assert all(≤(osizes[2R]), mt)
    @assert all(>(0), mt)

    for i in 1:sizes[1]
        for j in 1:sizes[1]
            @assert obasis[basis(RG)[i]*basis(RG)[j]] == mt[i, j]
        end
    end

    @assert maximum(mt) == osizes[2R]

    C = PropertyT._constraints(mt; augmented = true, id = 1)
    @assert length(C) == osizes[2R]
    @assert all(!isempty, C)
    mt
end
=#

function SymbolicWedderburn.WedderburnDecomposition(
    T::Type,
    G::Groups.Group,
    action::SymbolicWedderburn.Action,
    orbit_basis::OrbitBasis,
    basis_half,
    S = Rational{Int};
    semisimple = false,
)
    tbl = SymbolicWedderburn.CharacterTable(S, G)

    ehom = SymbolicWedderburn.CachedExtensionHomomorphism(
        G,
        action,
        basis_half;
        precompute = true,
    )

    Uπs = SymbolicWedderburn.symmetry_adapted_basis(
        T,
        tbl,
        ehom;
        semisimple = semisimple,
    )
    return SymbolicWedderburn.WedderburnDecomposition(
        basis_half,
        orbit_basis,
        Uπs,
        ehom,
    )
end

function sos_problem_primal(
    elt::StarAlgebras.AlgebraElement,
    order_unit::StarAlgebras.AlgebraElement,
    wedderburn::WedderburnDecomposition{B,<:OrbitBasis};
    upper_bound = Inf,
    augmented::Bool = iszero(StarAlgebras.aug(elt)) &&
        iszero(StarAlgebras.aug(order_unit)),
    check_orthogonality = true,
    show_progress = false,
    constraints = PropertyT.constraints(
        basis(wedderburn),
        invariant_vectors(wedderburn);
        augmented = augmented,
        basis_support = 1:length(basis(wedderburn)),
    ),
) where {B}
    @assert parent(elt) === parent(order_unit)
    if check_orthogonality
        if any(!isorth_projection, direct_summands(wedderburn))
            error(
                "Wedderburn decomposition contains a non-orthogonal projection",
            )
        end
    end

    prog = ProgressMeter.Progress(
        length(invariant_vectors(wedderburn));
        dt = 1,
        desc = "Adding constraints: ",
        enabled = show_progress,
        barlen = 60,
        showspeed = true,
    )

    feasibility_problem = iszero(order_unit)

    # problem creation starts here
    model = JuMP.Model()
    if !feasibility_problem # add λ of not?
        JuMP.@variable(model, λ)
        JuMP.@objective(model, Max, λ)
    end
    if isfinite(upper_bound)
        if feasibility_problem
            @warn "setting `upper_bound` with zero `unit` has no effect"
        else
            JuMP.@constraint(model, λ <= upper_bound)
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

    id_one = findfirst(isone, invariant_vectors(wedderburn))
    # adding linear constraints: one per orbit
    for (i, r) in enumerate(invariant_vectors(wedderburn))
        ProgressMeter.next!(prog; showvalues = __show_itrs(i, prog.n))
        augmented && i == id_one && continue
        A_r = sparse(constraints[r])

        Ms = SymbolicWedderburn.diagonalize!(
            Ms,
            A_r,
            wedderburn;
            trace_preserving = true,
        )
        for M in Ms
            SparseArrays.droptol!(M, _eps)
        end
        if r in basis(parent(elt))
            if feasibility_problem
                JuMP.@constraint(model, elt(r) == _dot(Ps, Ms))
            else
                JuMP.@constraint(
                    model,
                    elt(r) - λ * order_unit(r) == _dot(Ps, Ms)
                )
            end
        else
            JuMP.@constraint(model, 0 == PropertyT._dot(Ps, Ms))
        end
    end
    ProgressMeter.finish!(prog)
    return model, Ps
end
