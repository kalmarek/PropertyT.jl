include("Projections.jl")

###############################################################################
#
#  Iterator protocol for Nemo.FinField
#
###############################################################################

type FFEltsIter{T<:Nemo.FinField}
    all::Int
    field::T

    function FFEltsIter(F::T)
        return new(Int(characteristic(F)^degree(F)), F)
    end
end
FFEltsIter{T<:Nemo.FinField}(F::T) = FFEltsIter{T}(F)

import Base: start, next, done, eltype, length

Base.start(A::FFEltsIter) = (zero(A.field), 0)
Base.next(A::FFEltsIter, state) = next_ffelem(state...)
Base.done(A::FFEltsIter, state) = state[2] >= A.all
Base.eltype(::Type{FFEltsIter}) = elem_type(A.field)
Base.length(A::FFEltsIter) = A.all

function next_ffelem(f::Nemo.FinFieldElem, c::Int)
    if c == 0
        return (f, (f, 1))
    elseif c == 1
        f = one(parent(f))
        return (f, (f, 2))
    else
        f = gen(parent(f))*f
        return (f, (f, c+1))
    end
end

import Nemo.elements
elements(F::Nemo.FinField) = FFEltsIter(F)

###############################################################################
#
#  Orbit stuff
#
###############################################################################

function orbit_decomposition(G::Nemo.Group, E::Vector, rdict=GroupRings.reverse_dict(E))

    elts = collect(elements(G))

    tovisit = trues(E);
    orbits = Vector{Vector{Int}}()

    for i in 1:endof(E)
        if tovisit[i]
            orbit = Vector{Int}()
            a = E[i]
            for g in elts
                idx = rdict[g(a)]
                tovisit[idx] = false
                push!(orbit,idx)
            end
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

function orbit_constraint(constraints::Vector{Vector{Vector{Int64}}}, n)
    result = spzeros(n,n)
    for cnstr in constraints
        for p in cnstr
            result[p[2], p[1]] += 1.0
        end
    end
    return 1/length(constraints)*result
end

###############################################################################
#
#  Matrix- and C*-representations
#
###############################################################################

function matrix_repr(g::GroupElem, E, E_dict)
   rep_matrix = spzeros(Int, length(E), length(E))
   for (i,elt) in enumerate(E)
      j = E_dict[g(elt)]
      rep_matrix[i,j] = 1
   end
   return rep_matrix
end

function matrix_reps{T<:GroupElem}(G::Group, S::Vector{T}, AutS::Group, radius::Int)
   Id = (isa(G, Nemo.Ring) ? one(G) : G())
   E2, _ = Groups.generate_balls(S, Id, radius=radius)
   Edict = GroupRings.reverse_dict(E2)

   mreps = Dict(g=>matrix_repr(g, E2, Edict) for g in elements(AutS))
   return mreps
end

function reconstruct_sol{T<:GroupElem, S<:AbstractArray}(mreps::Dict{T, S},
    Us::Vector, Ps::Vector, dims::Vector)

    n = size(Us[1],1)
    recP = zeros(Float64, (n,n))
    Ust = transpose.(Us)
    for g in keys(mreps)
        A, B = mreps[g], mreps[inv(g)]
        for π in 1:length(Us)
            recP .+= sparsify(dims[π].* (A * Us[π]*Ps[π]*Ust[π] * B))
        end
    end
    recP .*= 1/length(keys(mreps))
    return recP
end

function Cstar_repr{T}(x::GroupRingElem{T}, mreps::Dict)
   res = zeros(size(mreps[first(keys(mreps))])...)

   for g in parent(x).basis
      if x[g] != zero(T)
         res .+= x[g].*mreps[g]
      end
   end

   return res
end

function orthSVD(M::AbstractMatrix)
    M = full(M)
    fact = svdfact(M)
    singv = fact[:S]
    M_rank = sum(singv .> maximum(size(M))*eps(eltype(singv)))
    return fact[:U][:,1:M_rank]
end

function compute_orbit_data{T<:GroupElem}(logger, name::String, G::Nemo.Group, S::Vector{T}, AutS; radius=2)
   isdir(name) || mkdir(name)

   info(logger, "Generating ball of radius $(2*radius)")

   # TODO: Fix that by multiple dispatch?
   Id = (isa(G, Nemo.Ring) ? one(G) : G())

   @time E4, sizes = Groups.generate_balls(S, Id, radius=2*radius);
   info(logger, "Balls of sizes $sizes.")
   info(logger, "Reverse dict")
   @time E_dict = GroupRings.reverse_dict(E4)

   info(logger, "Product matrix")
   @time pm = GroupRings.create_pm(E4, E_dict, sizes[radius], twisted=true)
   RG = GroupRing(G, E4, E_dict, pm)
   Δ = PropertyT.splaplacian(RG, S)
   @assert GroupRings.augmentation(Δ) == 0
   save(joinpath(name, "delta.jld"), "Δ", Δ.coeffs)
   save(joinpath(name, "pm.jld"), "pm", pm)

   info(logger, "Decomposing E into orbits of $(AutS)")
   @time orbs = orbit_decomposition(AutS, E4, E_dict)
   @assert sum(length(o) for o in orbs) == length(E4)
   save(joinpath(name, "orbits.jld"), "orbits", orbs)

   info(logger, "Action matrices")
   E2 = E4[1:sizes[radius]]
   @time AutS_mreps = Dict(g=>matrix_repr(g, E2, E_dict) for g in elements(AutS))

   info(logger, "Projections")
   @time AutS_mps = rankOne_projections(AutS);

   @time π_E_projections = [Cstar_repr(p, AutS_mreps) for p in AutS_mps]

   info(logger, "Uπs...")
   @time Uπs = orthSVD.(π_E_projections)

   multiplicities = size.(Uπs,2)
   info(logger, "multiplicities = $multiplicities")
   dimensions = [Int(p[AutS()]*Int(order(AutS))) for p in AutS_mps];
   info(logger, "dimensions = $dimensions")
   @assert dot(multiplicities, dimensions) == sizes[radius]

   save(joinpath(name, "U_pis.jld"), "Uπs", Uπs, "spUπs", sparsify.(Uπs), "dims", dimensions)
   return 0
end
