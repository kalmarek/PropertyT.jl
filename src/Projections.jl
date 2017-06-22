###############################################################################
#
#  Characters of PermutationGroup
#
###############################################################################

function chars(G::PermutationGroup)
   permtype_unsorted(σ::Nemo.perm) = [length(c) for c in cycles(σ)]
   permtype(σ::Nemo.perm) = sort(permtype_unsorted(σ))

   χ_id(σ::Nemo.perm) = 1

   χ_sgn(σ::Nemo.perm) = (-1)^parity(σ)

   function χ_reg(σ::Nemo.perm)
      fixed_points = countnz([(x == y? 1 : 0) for (x,y) in enumerate(σ.d)])
      return fixed_points - 1
   end

   χ_regsgn(σ::Nemo.perm) = (-1)^parity(σ)*χ_reg(σ)

   function χ_regviaS3(σ::Nemo.perm)
      @assert parent(σ).n == 4
      t = permtype(σ)
         if t == [1,1,1,1]
            result = 2
         elseif t == [2,2]
            result = 2
         elseif t == [1,3]
            result = -1
         else
            result = 0
         end
      return result
   end

   chars = [χ_id, χ_sgn, χ_regviaS3, χ_reg, χ_regsgn]

   if G.n == 1
      return chars[1:1]
   elseif G.n == 2
      return chars[1:2]
   elseif G.n == 3
      return [chars[1:2]..., chars[4]]
   elseif G.n == 4
      return chars[1:5]
   else
      throw("Characters for $G unknown!")
   end
end

###############################################################################
#
#  Character of DirectProducts
#
###############################################################################

function epsilon(i, g::DirectProductGroupElem)
    return reduce(*, 1, ((-1)^isone(g.elts[j]) for j in 1:i))
end

###############################################################################
#
#  Projections
#
###############################################################################

function central_projection(RG::GroupRing, char::Function, T::Type=Rational{Int})
    result = RG(T)
    for g in RG.basis
        result[g] = char(inv(g))
    end
    return convert(T, char(RG.group())//Int(order(RG.group))*result)
end

function rankOne_projections(G::PermutationGroup, T::Type=Rational{Int})
   RG = GroupRing(G)
   projections = [central_projection(RG, χ, T) for χ in chars(G)]

   if G.n == 1 || G.n == 2
      return  projections
   elseif G.n == 3
      rankone_projs = [
         projections[1],
         projections[2],
         1//2*(one(RG) - RG(RG.group([2,1,3])))*projections[3]
         ]
      return rankone_projs
   elseif G.n == 4
      rankone_projs = [
         projections[1],
         projections[2],
         1//2*(one(RG) - RG(RG.group([2,1,3,4])))*projections[3],
         1//2*(one(RG) - RG(RG.group([2,1,3,4])))*projections[4],
         1//2*(one(RG) + RG(RG.group([2,1,3,4])))*projections[5]]
      return rankone_projs
   else
      throw("Rank-one projections for $G unknown!")
   end
end

function rankOne_projections(BN::WreathProduct, T::Type=Rational{Int})

   N = BN.P.n
    # projections as elements of the group rings RSₙ
   SNprojs_nc = [rankOne_projections(PermutationGroup(i), T) for i in 1:N]

   # embedding into group ring of BN
   RBN = GroupRing(BN)
   RFFFF_projs = [central_projection(GroupRing(BN.N), g->epsilon(i,g), T)
        for i in 0:BN.P.n]
   Qs = [RBN(q, g -> BN(g)) for q in RFFFF_projs]

   function incl(k::Int, g::perm, WP::WreathProduct=BN)
      @assert length(g.d) + k <= WP.P.n
      arr = [1:k; g.d .+ k; (length(g.d)+k+1):WP.P.n]
      return WP(WP.P(arr))
   end

    all_projs=[Qs[1]*RBN(p, g-> incl(0,g)) for p in SNprojs_nc[N]]

   for i in 1:N-1
        Sk_first = [RBN(p, g->incl(0,g)) for p in SNprojs_nc[i]]
        Sk_last  = [RBN(p, g->incl(i,g)) for p in SNprojs_nc[N-i]]
        append!(all_projs,
            [Qs[i+1]*p1*p2 for (p1,p2) in Base.product(Sk_first,Sk_last)])
   end

   append!(all_projs, [Qs[N+1]*RBN(p, g-> incl(0,g)) for p in SNprojs_nc[N]])

   return all_projs
end
