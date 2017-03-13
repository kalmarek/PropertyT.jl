using JLD
using JuMP
import Primes: isprime
import SCS: SCSSolver
import Mosek: MosekSolver

using Mods

using Groups

function SL_generatingset(n::Int)

    indexing = [(i,j) for i in 1:n for j in 1:n if i≠j]

    S = [E(i,j,N=n) for (i,j) in indexing];
    S = vcat(S, [convert(Array{Int,2},x') for x in S]);
    S = vcat(S, [convert(Array{Int,2},inv(x)) for x in S]);
    return unique(S)
end

function E(i::Int, j::Int; val=1, N::Int=3, mod=Inf)
   @assert i≠j
   m = eye(Int, N)
   m[i,j] = val
   if mod == Inf
       return m
   else
       return [Mod(x,mod) for x in m]
   end
end

function cofactor(i,j,M)
    z1 = ones(Bool,size(M,1))
    z1[i] = false

    z2 = ones(Bool,size(M,2))
    z2[j] = false

    return M[z1,z2]
end

import Base.LinAlg.det

function det(M::Array{Mod,2})
    if size(M,1) ≠ size(M,2)
        d = Mod(0,M[1,1].mod)
    elseif size(M,1) == 2
        d =  M[1,1]*M[2,2] - M[1,2]*M[2,1]
    else
        d = zero(eltype(M))
        for i in 1:size(M,1)
            d += (-1)^(i+1)*M[i,1]*det(cofactor(i,1,M))
        end
    end
#     @show (M, d)
    return d
end

function adjugate(M)
    K = similar(M)
    for i in 1:size(M,1), j in 1:size(M,2)
        K[j,i] = (-1)^(i+j)*det(cofactor(i,j,M))
    end
    return K
end

import Base: inv, one, zero, *

one(::Type{Mod}) = 1
zero(::Type{Mod}) = 0
zero(x::Mod) = Mod(x.mod)

function inv(M::Array{Mod,2})
    d = det(M)
    d ≠ 0*d || thow(ArgumentError("Matrix is not invertible!"))
    return inv(det(M))*adjugate(M)
    return adjugate(M)
end

function SL_generatingset(n::Int, p::Int)
    (p > 1 && n > 1) || throw(ArgumentError("Both n and p should be integers!"))
    isprime(p) || throw(ArgumentError("p should be a prime number!"))

    indexing = [(i,j) for i in 1:n for j in 1:n if i≠j]
    S = [E(i,j, N=n, mod=p) for (i,j) in indexing]
    S = vcat(S, [inv(s) for s in S])
    S = vcat(S, [permutedims(x, [2,1]) for x in S]);

    return unique(S)
end

function products{T}(U::AbstractVector{T}, V::AbstractVector{T})
    result = Vector{T}()
    for u in U
        for v in V
            push!(result, u*v)
        end
    end
    return unique(result)
end

function ΔandSDPconstraints(identity, S)
    B₁ = vcat([identity], S)
    B₂ = products(B₁, B₁);
    B₃ = products(B₁, B₂);
    B₄ = products(B₁, B₃);
    @assert B₄[1:length(B₂)] == B₂

    product_matrix = create_product_matrix(B₄,length(B₂));
    sdp_constraints = constraints_from_pm(product_matrix, length(B₄))
    L_coeff = splaplacian_coeff(S, B₂, length(B₄));
    Δ = GroupAlgebraElement(L_coeff, product_matrix)

    return Δ, sdp_constraints
end






@everywhere push!(LOAD_PATH, "./")
using GroupAlgebras
include("property(T).jl")

const N = 3

const name = "SL$(N)Z"
const ID = eye(Int, N)
S() = SL_generatingset(N)
const upper_bound=0.27


# const p = 7
# const upper_bound=0.738 # (N,p) = (3,7)

# const name = "SL($N,$p)"
# const ID = [Mod(x,p) for x in eye(Int,N)]
# S() = SL_generatingset(N, p)

@time check_property_T(name, ID, S; verbose=true, tol=1e-10, upper_bound=upper_bound)
