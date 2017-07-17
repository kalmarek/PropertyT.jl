###############################################################################
#
#  Characters of PermutationGroup
#
###############################################################################

function chars(G::PermutationGroup)
   permtype_unsorted(σ::Nemo.perm) = [length(c) for c in cycles(σ)]
   permtype(σ::Nemo.perm) = sort(permtype_unsorted(σ), rev=true)

   χ_id(σ::Nemo.perm) = 1

   χ_sgn(σ::Nemo.perm) = sign(σ)

   function χ_reg(σ::Nemo.perm)
      fixed_points = countnz([(x == y? 1 : 0) for (x,y) in enumerate(σ.d)])
      return fixed_points - 1
   end

   χ_regsgn(σ::Nemo.perm) = sign(σ)*χ_reg(σ)

   if G.n == 1
      return [χ_id]

   elseif G.n == 2
      return [χ_id, χ_sgn]

   elseif G.n == 3
      return [χ_id, χ_sgn, χ_reg]

   elseif G.n == 4

      function χ_regviaS3(σ::Nemo.perm)
         vals = Dict{Vector{Int}, Int}(
            [1,1,1,1] => 2,
            [2,1,1]   => 0,
            [2,2]     => 2,
            [3,1]     =>-1,
            [4]       => 0
         )
         return vals[permtype(σ)]
      end

      return [χ_id, χ_sgn, χ_regviaS3, χ_reg, χ_regsgn]

   elseif G.n == 5

      function ϱ(σ::Nemo.perm)
         vals = Dict{Vector{Int}, Int}(
            [1,1,1,1,1] => 5,
            [2,1,1,1]   => 1,
            [2,2,1]     => 1,
            [3,1,1]     =>-1,
            [3,2]       => 1,
            [4,1]       =>-1,
            [5]         => 0
         )
         return vals[permtype(σ)]
      end

      ϱ_sgn(σ::Nemo.perm) = sign(σ)*ϱ(σ)

      function ψ(σ::Nemo.perm)
         vals = Dict{Vector{Int}, Int}(
            [1,1,1,1,1] => 6,
            [2,1,1,1]   => 0,
            [2,2,1]     => -2,
            [3,1,1]     => 0,
            [3,2]       => 0,
            [4,1]       => 0,
            [5]         => 1
         )
         return vals[permtype(σ)]
      end

      return [χ_id, χ_sgn, χ_reg, χ_regsgn, ϱ, ϱ_sgn, ψ]
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

function central_projection{F<:Function}(RG::GroupRing, chi::F, T::Type=Rational{Int})
    result = RG(T)
    result.coeffs = full(result.coeffs)
    for g in RG.basis
        result[g] = chi(g)
    end
    dim = result[RG.group()]
    ord = Int(order(RG.group))
    return convert(T, (dim//ord))*result
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
