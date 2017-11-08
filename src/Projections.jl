###############################################################################
#
#  Characters of Symmetric Group and DirectProduct
#
###############################################################################

abstract AbstractCharacter

immutable PermCharacter <: AbstractCharacter
   p::Partition
end

immutable DirectProdCharacter <: AbstractCharacter
   i::Int
end

function (chi::PermCharacter)(g::Nemo.perm)
   R = Nemo.partitionseq(chi.p)
   p = Partition(Nemo.permtype(g))
   return Int(Nemo.MN1inner(R, p, 1, Nemo._charvalsTable))
end

Nemo.isone(p::GroupElem) = p == parent(p)()

## NOTE: this works only for Z/2!!!!
function (chi::DirectProdCharacter)(g::DirectProductGroupElem)
   return reduce(*, 1, ((-1)^isone(g.elts[j]) for j in 1:chi.i))
end

for T in [PermCharacter, DirectProdCharacter]
   @eval begin
      function (chi::$T)(X::GroupRingElem)
         RG = parent(X)
         z = zero(eltype(X))
         result = z
         for i in 1:length(X.coeffs)
            if X.coeffs[i] != z
               result += chi(RG.basis[i])*X.coeffs[i]
            end
         end
         return result
      end
   end
end

###############################################################################
#
#  Projections
#
###############################################################################

function central_projection(RG::GroupRing, chi::AbstractCharacter, T::Type=Rational{Int})
   result = RG(T)
   result.coeffs = full(result.coeffs)
   dim = chi(RG.group())
   ord = Int(order(RG.group))

   for g in RG.basis
      result[g] = convert(T, (dim//ord)*chi(g))
   end

   return result
end

function idempotents(RG::GroupRing{PermGroup}, T::Type=Rational{Int})
   if RG.group.n == 1
      return GroupRingElem{T}[one(RG,T)]
   elseif RG.group.n == 2
      Id = one(RG,T)
      transp = convert(T, RG(RG.group([2,1])))
      return GroupRingElem{T}[1//2*(Id + transp), 1//2*(Id - transp)]
   end
   projs = Vector{Vector{perm}}()
   for l in 2:RG.group.n
      u = RG.group([circshift([i for i in 1:l], -1); [i for i in l+1:RG.group.n]])
      i = 0
      while (l-1)*i <= RG.group.n
         v = RG.group(circshift(collect(1:RG.group.n), i))
         k = inv(v)*u*v
         push!(projs, generateGroup([k], RG.group.n))
         i += 1
      end
   end

   idems = Vector{GroupRingElem{T}}()
   for p in projs
      append!(idems, [RG(p, T), RG(p, T, alt=true)])
   end

   return unique(idems)
end

function rankOne_projection{T}(chi::PropertyT.PermCharacter, idems::Vector{GroupRingElem{T}})

   RG = parent(first(idems))

   ids = [[one(RG, T)]; idems]

   for (i,j,k) in Base.product(ids, ids, ids)
      if chi(i) == zero(T) || chi(j) == zero(T) || chi(k) == zero(T)
         continue
      end
      elt = i*j*k
      elt^2 == elt || continue
      if chi(elt) == one(T)
         return elt
         # return (i,j,k)
      end
   end
   throw("Couldn't find rank-one projection for $chi")
end

function rankOne_projections(G::PermutationGroup, T::Type=Rational{Int})
   if G.n == 1
      return [one(GroupRing(G), T)]
   elseif G.n < 8
      RG = GroupRing(G, fastm=true)
   else
      RG = GroupRing(G, fastm=false)
   end

   RGidems = idempotents(RG, T)
   l = length(Partitions(G.n))

   parts = collect(Partitions(G.n))
   chars = [PropertyT.PermCharacter(p) for p in parts]
   min_projs = Vector{eltype(RGidems)}(l)

   for i in 1:l
      chi = PropertyT.PermCharacter(parts[i])
      min_projs[i] = rankOne_projection(chi,RGidems)*central_projection(RG,chi)
   end

   return min_projs
end

function rankOne_projections(BN::WreathProduct, T::Type=Rational{Int})

   N = BN.P.n
   # projections as elements of the group rings RSₙ
   SNprojs_nc = [rankOne_projections(PermutationGroup(i)) for i in 1:N]

   # embedding into group ring of BN
   RBN = GroupRing(BN)
   RFFFF_projs = [central_projection(GroupRing(BN.N), DirectProdCharacter(i),T)
      for i in 1:BN.P.n]

   e0 = central_projection(GroupRing(BN.N), DirectProdCharacter(0), T)
   Q0 = RBN(e0, g -> BN(g))
   Qs = [RBN(q, g -> BN(g)) for q in RFFFF_projs]

   all_projs = [Q0*RBN(p, g->BN(g)) for p in SNprojs_nc[N]]

   range = collect(1:N)
   for i in 1:N-1
      first_emb = g->BN(Nemo.emb!(BN.P(), g, range[1:i]))
      last_emb = g->BN(Nemo.emb!(BN.P(), g, range[i+1:end]))

      Sk_first = [RBN(p, first_emb) for p in SNprojs_nc[i]]
      Sk_last = [RBN(p, last_emb) for p in SNprojs_nc[N-i]]

      append!(all_projs,
      [Qs[i]*p1*p2 for (p1,p2) in Base.product(Sk_first,Sk_last)])
   end

   append!(all_projs, [Qs[N]*RBN(p, g->BN(g)) for p in SNprojs_nc[N]])

   return all_projs
end

##############################################################################
#
#   General Groups Misc
#
##############################################################################

doc"""
    products(X::Vector{GroupElem}, Y::Vector{GroupElem}, op=*)
> Returns a vector of all possible products (or `op(x,y)`), where $x\in X$ and
> $y\in Y$ are group elements. You may specify which operation is used when
> forming 'products' by adding `op` (which is `*` by default).
"""
function products{T<:GroupElem}(X::AbstractVector{T}, Y::AbstractVector{T}, op=*)
   result = Vector{T}()
   seen = Set{T}()
   for x in X
      for y in Y
         z = op(x,y)
         if !in(z, seen)
            push!(seen, z)
            push!(result, z)
         end
      end
   end
   return result
end

doc"""
    generateGroup(gens::Vector{GroupElem}, r=2, Id=parent(first(gens))(), op=*)
> Produces all elements of a group generated by elements in `gens` in ball of
> radius `r` (word-length metric induced by `gens`).
> If `r(=2)` is specified the procedure will terminate after generating ball
> of radius `r` in the word-length metric induced by `gens`.
> The identity element `Id` and binary operation function `op` can be supplied
> to e.g. take advantage of additive group structure.
"""
function generateGroup{T<:GroupElem}(gens::Vector{T}, r=2, Id::T=parent(first(gens))(), op=*)
   n = 0
   R = 1
   elts = gens
   gens = [Id; gens]
   while n ≠ length(elts) && R < r
      # @show elts
      R += 1
      n = length(elts)
      elts = products(elts, gens, op)
   end
   return elts
end
