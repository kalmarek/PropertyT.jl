include("Projections.jl")

###############################################################################
#
#  Orbit stuff
#
###############################################################################

function orbit_decomposition(G::Group, E::Vector, rdict=GroupRings.reverse_dict(E))

    elts = collect(elements(G))

    tovisit = trues(E);
    orbits = Vector{Vector{Int}}()

    orbit = zeros(Int, length(elts))

    for i in 1:endof(E)
        if tovisit[i]
            g = E[i]
            Threads.@threads for j in 1:length(elts)
                orbit[j] = rdict[elts[j](g)]
            end
            tovisit[orbit] = false
            push!(orbits, unique(orbit))
        end
    end
    return orbits
end

function orbit_spvector(vect::AbstractVector, orbits)
    orb_vector = spzeros(length(orbits))

    for (i,o) in enumerate(orbits)
        k = vect[collect(o)]
        val = k[1]
        @assert all(k .== val)
        orb_vector[i] = val
    end

    return orb_vector
end

function orbit_constraint(constraints::Vector{Vector{Int}}, n)
    result = spzeros(n,n)
    for cnstr in constraints
        result[cnstr] += 1.0/length(constraints)
    end
    return result
end

###############################################################################
#
#  Matrix-, Permutation- and C*-representations
#
###############################################################################

function matrix_repr(p::perm)
    N = parent(p).n
    return sparse(1:N, p.d, [1.0 for _ in 1:N])
end

function matrix_reps(preps::Dict{T,perm{I}}) where {T<:GroupElem, I<:Integer}
    kk = collect(keys(preps))
    mreps = Vector{SparseMatrixCSC{Float64, Int}}(length(kk))
    Threads.@threads for i in 1:length(kk)
        mreps[i] = matrix_repr(preps[kk[i]])
    end
    return Dict(kk[i] => mreps[i] for i in 1:length(kk))
end

function perm_repr(g::GroupElem, E::Vector, E_dict)
    p = Vector{Int}(length(E))
    for (i,elt) in enumerate(E)
        p[i] = E_dict[g(elt)]
    end
    return p
end

function perm_reps(G::Group, E::Vector, E_rdict=GroupRings.reverse_dict(E))
    elts = collect(elements(G))
    l = length(elts)
    preps = Vector{perm}(l)

    permG = PermutationGroup(length(E))

    Threads.@threads for i in 1:l
        preps[i] = permG(PropertyT.perm_repr(elts[i], E, E_rdict), false)
    end

    return Dict(elts[i]=>preps[i] for i in 1:l)
end

function perm_reps(S::Vector, autS::Group, radius::Int)
    E, _ = Groups.generate_balls(S, radius=radius)
    return perm_reps(autS, E)
end

function reconstruct_sol(preps::Dict{T, S}, Us::Vector, Ps::Vector, dims::Vector) where {T<:GroupElem, S<:perm}

    l = length(Us)
    transfP = [dims[π].*Us[π]*Ps[π]*Us[π]' for π in 1:l]
    tmp = [zeros(Float64, size(first(transfP))) for _ in 1:l]
    perms = collect(keys(preps))

    @inbounds Threads.@threads for π in 1:l
        for p in perms
            BLAS.axpy!(1.0, view(transfP[π], preps[p].d, preps[p].d), tmp[π])
        end
    end

    recP = 1/length(perms) .* sum(tmp)
    for i in eachindex(recP)
        if abs(recP[i]) .< eps(eltype(recP))*100
            recP[i] = zero(eltype(recP))
        end
    end
    return recP
end

function Cstar_repr(x::GroupRingElem{T}, mreps::Dict) where {T}
    return sum(x[i].*mreps[parent(x).basis[i]] for i in findn(x.coeffs))
end

function orthSVD{T}(M::AbstractMatrix{T})
    M = full(M)
    fact = svdfact(M)
    M_rank = sum(fact[:S] .> maximum(size(M))*eps(T))
    return fact[:U][:,1:M_rank]
end

function compute_orbit_data{T<:GroupElem}(name::String, S::Vector{T}, autS::Group; radius=2)
    isdir(name) || mkdir(name)

    info("Generating ball of radius $(2*radius)")

    # TODO: Fix that by multiple dispatch?
    G = parent(first(S))
    Id = (isa(G, Ring) ? one(G) : G())

    @time E_2R, sizes = Groups.generate_balls(S, Id, radius=2*radius);
    info("Balls of sizes $sizes.")
    info("Reverse dict")
    @time E_rdict = GroupRings.reverse_dict(E_2R)

    info("Product matrix")
    @time pm = GroupRings.create_pm(E_2R, E_rdict, sizes[radius], twisted=true)
    RG = GroupRing(G, E_2R, E_rdict, pm)
    Δ = PropertyT.spLaplacian(RG, S)
    @assert GroupRings.aug(Δ) == 0
    save(filename(name, :Δ), "Δ", Δ.coeffs)
    save(filename(name, :pm), "pm", pm)

    info("Decomposing E into orbits of $(autS)")
    @time orbs = orbit_decomposition(autS, E_2R, E_rdict)
    @assert sum(length(o) for o in orbs) == length(E_2R)
    info("E consists of $(length(orbs)) orbits!")
    save(joinpath(name, "orbits.jld"), "orbits", orbs)

    info("Action matrices")
    @time reps = perm_reps(autS, E_2R[1:sizes[radius]], E_rdict)
    save_preps(filename(name, :preps), reps)
    reps = matrix_reps(reps)

    info("Projections")
    @time autS_mps = Projections.rankOne_projections(GroupRing(autS));

    @time π_E_projections = [Cstar_repr(p, reps) for p in autS_mps]

    info("Uπs...")
    @time Uπs = orthSVD.(π_E_projections)

    multiplicities = size.(Uπs,2)
    info("multiplicities = $multiplicities")
    dimensions = [Int(p[autS()]*Int(order(autS))) for p in autS_mps];
    info("dimensions = $dimensions")
    @assert dot(multiplicities, dimensions) == sizes[radius]

    save(joinpath(name, "U_pis.jld"),
    "Uπs", Uπs,
    "dims", dimensions)
    return 0
end
