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

function central_projection(RG::GroupRing, chi::Function, T::Type=Rational{Int})
    result = RG(T)
    result.coeffs = full(result.coeffs)
    dim = chi(RG.group())
    ord = Int(order(RG.group))

    for g in RG.basis
        result[g] = convert(T, (dim//ord)*chi(g))
    end

    return result
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
         1//2*(one(RG, T) - RG(G([2,1,3]), T))*projections[3]
      ]
      return rankone_projs
   elseif G.n == 4
      rankone_projs = [
         projections[1],
         projections[2],
         1//2*(one(RG, T) - RG(G([2,1,3,4]), T))*projections[3],
         1//2*(one(RG, T) - RG(G([2,1,3,4]), T))*projections[4],
         1//2*(one(RG, T) + RG(G([2,1,3,4]), T))*projections[5]
      ]
      return rankone_projs
   elseif G.n == 5
      p⁺ = 1//2*(one(RG, T) + RG(G([2,1,3,4,5]), T))
      p⁻ = 1//2*(one(RG, T) - RG(G([2,1,3,4,5]), T))

      q⁺ = 1//2*(one(RG, T) + RG(G([1,2,4,3,5]), T))
      q⁻ = 1//2*(one(RG, T) - RG(G([1,2,4,3,5]), T))

      rankone_projs = [
         projections[1],
         projections[2],
         p⁻*projections[3],
         p⁺*projections[4],
         p⁺*q⁺*projections[5],
         p⁻*q⁻*projections[6],
         p⁺*q⁺*projections[7]
      ]
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
