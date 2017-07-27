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
        result[g] = convert(T, (Int(dim)//ord)*Int(chi(g)))
    end

    return result
end

function rankOne_projections(G::PermutationGroup, T::Type=Rational{Int})
   RG = GroupRing(G)
   cprojs = [central_projection(RG, χ, T) for χ in (character(λ) for λ in Partitions(G.n))]

   if G.n == 1 || G.n == 2
      return  cprojs
   elseif G.n == 3
      p = 1//2*(one(RG, T) - RG(G([2,1,3]), T))
      rankone_projs = [
         cprojs[1],        # alternating
         p*cprojs[2],      # regular
         cprojs[3]         # trivial
      ]
   elseif G.n == 4
       p⁺ = 1//2*(one(RG, T) + RG(G([2,1,3,4]), T))
       p⁻ = 1//2*(one(RG, T) - RG(G([2,1,3,4]), T))
       rankone_projs = [
         cprojs[1],        # alternating
         p⁺*cprojs[2],     # alt_regular
         p⁻*cprojs[3],     # regular
         p⁻*cprojs[4],     # via projection to S₃
         cprojs[5]         # trivial
      ]
   elseif G.n == 5
      p⁺ = 1//2*(one(RG, T) + RG(G([2,1,3,4,5]), T))
      p⁻ = 1//2*(one(RG, T) - RG(G([2,1,3,4,5]), T))

      q⁺ = 1//2*(one(RG, T) + RG(G([1,2,4,3,5]), T))
      q⁻ = 1//2*(one(RG, T) - RG(G([1,2,4,3,5]), T))

      rankone_projs = [
         cprojs[1],        # alternating
         p⁺*cprojs[2],     # alt_regular
         p⁺*q⁺*cprojs[3],  # ψ
         p⁺*q⁺*cprojs[4],  # alt_ϱ
         p⁻*cprojs[5],     # regular
         p⁻*q⁻*cprojs[6],  # ϱ
         cprojs[7]         # trivial
      ]
   else
      throw("Rank-one projections for $G unknown!")
   end
   return rankone_projs
end

function rankOne_projections(BN::WreathProduct, T::Type=Rational{Int})

   N = BN.P.n
   # projections as elements of the group rings RSₙ
   SNprojs_nc = [rankOne_projections(PermutationGroup(i), T) for i in 1:N]

   # embedding into group ring of BN
   RBN = GroupRing(BN)
   RFFFF_projs = [
      central_projection(GroupRing(BN.N), g->epsilon(i,g), T) for i in 1:BN.P.n
      ]

   e0 = central_projection(GroupRing(BN.N), g->epsilon(0,g), T)
   Q0 = RBN(e0, g -> BN(g))
   Qs = [RBN(q, g -> BN(g)) for q in RFFFF_projs]

   all_projs = [Q0*RBN(p, g->BN(g)) for p in SNprojs_nc[N]]

   range = collect(1:N)
   for i in 1:N-1

      Sk_first = [RBN(p, g->BN(Nemo.emb!(BN.P(), g, range[1:i]))) for p in SNprojs_nc[i]]
      Sk_last = [RBN(p, g->BN(Nemo.emb!(BN.P(), g, range[i+1:end]))) for p in SNprojs_nc[N-i]]

      append!(all_projs,
      [Qs[i]*p1*p2 for (p1,p2) in Base.product(Sk_first,Sk_last)])
   end

   append!(all_projs, [Qs[N]*RBN(p, g->BN(g)) for p in SNprojs_nc[N]])

   return all_projs
end
