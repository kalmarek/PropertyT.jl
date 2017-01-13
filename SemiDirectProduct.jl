module SemiDirectProduct

import Base: convert, show, isequal, ==, size, inv
import Base: +, -, *, //

export SemiDirectProductElement, matrix_repr

"""
Implements elements of a semidirect product of groups H and N, where N is normal in the product. Usually written as H ⋉ N.
The multiplication inside semidirect product is defined as
    (h₁, n₁) ⋅ (h₂, n₂) = (h₁h₂, n₁φ(h₁)(n₂)),
where φ:H → Aut(N) is a homomorphism.

In the case below we implement H = GL(n,K) and N = Kⁿ, the Affine Group (i.e. GL(n,K) ⋉ Kⁿ) where elements of GL(n,K) act on vectors in Kⁿ via matrix multiplication.
# Arguments:
* `h::Array{T,2}` : square invertible matrix (element of GL(n,K))
* `n::Vector{T,1}` : vector in Kⁿ
* `φ = φ(h,n) = φ(h)(n)` :2-argument function which defines the action of GL(n,K) on Kⁿ; matrix-vector multiplication by default.
"""
immutable SemiDirectProductElement{T<:Number}
    h::Array{T,2}
    n::Vector{T}
    φ::Function

    function SemiDirectProductElement(h::Array{T,2},n::Vector{T},φ::Function)
        # size(h,1) == size(h,2)|| throw(ArgumentError("h has to be square matrix"))
        det(h) ≠ 0 || throw(ArgumentError("h has to be invertible!"))
        new(h,n,φ)
    end
end

SemiDirectProductElement{T}(h::Array{T,2}, n::Vector{T}, φ) =
    SemiDirectProductElement{T}(h,n,φ)

SemiDirectProductElement{T}(h::Array{T,2}, n::Vector{T}) =
    SemiDirectProductElement(h,n,*)

SemiDirectProductElement{T}(h::Array{T,2}) =
    SemiDirectProductElement(h,zeros(h[:,1]))

SemiDirectProductElement{T}(n::Vector{T}) =
        SemiDirectProductElement(eye(eltype(n), n))

convert{T<:Number}(::Type{T}, X::SemiDirectProductElement) =
    SemiDirectProductElement(convert(Array{T,2},X.h),
                             convert(Vector{T},X.n),
                             X.φ)

size(X::SemiDirectProductElement) = (size(X.h), size(X.n))

matrix_repr{T}(X::SemiDirectProductElement{T}) =
    [X.h X.n; zeros(T, 1, size(X.h,2)) [1]]

show{T}(io::IO, X::SemiDirectProductElement{T}) = print(io,
    "Element of SemiDirectProduct over $T of size $(size(X)):\n",
    matrix_repr(X))

function isequal{T}(X::SemiDirectProductElement{T}, Y::SemiDirectProductElement{T})
    X.h == Y.h || return false
    X.n == Y.n || return false
    X.φ == Y.φ || return false
    return true
end

function isequal{T,S}(X::SemiDirectProductElement{T}, Y::SemiDirectProductElement{S})
    W = promote_type(T,S)
    warn("Comparing elements with different coefficients! trying to promoting to $W")
    X = convert(W, X)
    Y = convert(W, Y)
    return isequal(X,Y)
end

(==)(X::SemiDirectProductElement, Y::SemiDirectProductElement) = isequal(X, Y)

function semidirect_multiplication{T}(X::SemiDirectProductElement{T},
                                      Y::SemiDirectProductElement{T})
    size(X) == size(Y) || throw(ArgumentError("trying to multiply elements from different groups!"))
    return SemiDirectProductElement(X.h*Y.h, X.n + X.φ(X.h, Y.n))
end

(*){T}(X::SemiDirectProductElement{T}, Y::SemiDirectProductElement{T}) =
    semidirect_multiplication(X,Y)

inv{T}(X::SemiDirectProductElement{T}) =
    SemiDirectProductElement(inv(X.h), X.φ(inv(X.h), -X.n))


end
