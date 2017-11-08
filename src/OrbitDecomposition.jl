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
            orbit = zeros(Int, length(elts))
            a = E[i]
            Threads.@threads for i in 1:length(elts)
               orbit[i] = rdict[elts[i](a)]
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

function orbit_constraint(constraints::Vector{Vector{Tuple{Int,Int}}}, n)
    result = spzeros(n,n)
    for cnstr in constraints
        for p in cnstr
            result[p[2], p[1]] += 1.0/length(constraints)
        end
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

function matrix_reps{T<:GroupElem}(preps::Dict{T,perm})
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
   preps = Vector{Nemo.perm}(l)

   permG = Nemo.PermutationGroup(length(E))

   Threads.@threads for i in 1:l
      preps[i] = permG(PropertyT.perm_repr(elts[i], E, E_rdict))
   end

   return Dict(elts[i]=>preps[i] for i in 1:l)
end

function perm_reps(S::Vector, AutS::Group, radius::Int)
   E, _ = Groups.generate_balls(S, radius=radius)
   return perm_reps(AutS, E)
end

function reconstruct_sol{T<:GroupElem, S<:Nemo.perm}(preps::Dict{T, S},
   aUs::Vector, aPs::Vector, adims::Vector)

   idx = [π for π in 1:length(aUs) if size(aUs[π], 2) != 0]
   Us = aUs[idx]
   Ps = aPs[idx]
   dims = adims[idx];

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
   recP[abs.(recP) .< eps(eltype(recP))] = zero(eltype(recP))
   return recP
end

function Cstar_repr{T}(x::GroupRingElem{T}, mreps::Dict)
   return sum(x[g].*mreps[g] for g in parent(x).basis if x[g] != zero(T))
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

   @logtime logger E4, sizes = Groups.generate_balls(S, Id, radius=2*radius);
   info(logger, "Balls of sizes $sizes.")
   info(logger, "Reverse dict")
   @logtime logger E_dict = GroupRings.reverse_dict(E4)

   info(logger, "Product matrix")
   @logtime logger pm = GroupRings.create_pm(E4, E_dict, sizes[radius], twisted=true)
   RG = GroupRing(G, E4, E_dict, pm)
   Δ = PropertyT.splaplacian(RG, S)
   @assert GroupRings.augmentation(Δ) == 0
   save(joinpath(name, "delta.jld"), "Δ", Δ.coeffs)
   save(joinpath(name, "pm.jld"), "pm", pm)

   info(logger, "Decomposing E into orbits of $(AutS)")
   @logtime logger orbs = orbit_decomposition(AutS, E4, E_dict)
   @assert sum(length(o) for o in orbs) == length(E4)
   save(joinpath(name, "orbits.jld"), "orbits", orbs)

   info(logger, "Action matrices")
   @logtime logger reps = perm_reps(AutS, E_2R[1:sizes[radius]], E_rdict)
   save_preps(joinpath(name, "preps.jld"), reps)
   reps = matrix_reps(reps)

   info(logger, "Projections")
   @logtime logger AutS_mps = rankOne_projections(AutS);

   @logtime logger π_E_projections = [Cstar_repr(p, AutS_mreps) for p in AutS_mps]

   info(logger, "Uπs...")
   @logtime logger Uπs = orthSVD.(π_E_projections)

   multiplicities = size.(Uπs,2)
   info(logger, "multiplicities = $multiplicities")
   dimensions = [Int(p[AutS()]*Int(order(AutS))) for p in AutS_mps];
   info(logger, "dimensions = $dimensions")
   @assert dot(multiplicities, dimensions) == sizes[radius]

   save(joinpath(name, "U_pis.jld"),
         "Uπs", Uπs,
         "spUπs", sparsify!.(deepcopy(Uπs), check=true, verbose=true),
         "dims", dimensions)
   return 0
end
