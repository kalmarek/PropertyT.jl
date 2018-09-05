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

function Cstar_repr(x::GroupRingElem{T}, mreps::Dict) where {T}
    nzeros = findn(x.coeffs)
    return sum(x[i].*mreps[parent(x).basis[i]] for i in nzeros)
end

function orthSVD(M::AbstractMatrix{T}) where {T<:AbstractFloat}
    M = full(M)
    fact = svdfact(M)
    M_rank = sum(fact[:S] .> maximum(size(M))*eps(T))
    return fact[:U][:,1:M_rank]
end

end
