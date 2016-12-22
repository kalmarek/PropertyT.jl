module GroupAlgebras

import Base: convert, show, isequal, ==
import Base: +, -, *, //
import Base: size, length, norm

export GroupAlgebraElement

typealias CoordinateVector{T<:Number} AbstractVector{T}

immutable GroupAlgebraElement{T<:CoordinateVector}
    coordinates::T
    product_matrix::Array{Int,2}
    # basis::Array{Any,1}

    function GroupAlgebraElement(coordinates::T,
        product_matrix::Array{Int,2})

        size(product_matrix, 1) == size(product_matrix, 2) ||
            throw(ArgumentError("Product matrix has to be square"))
        new(coordinates, product_matrix)
    end
end

# GroupAlgebraElement(c,pm,b) = GroupAlgebraElement(c,pm)
GroupAlgebraElement{T}(c::T,pm) = GroupAlgebraElement{T}(c,pm)

convert{T<:Number}(::Type{T}, X::GroupAlgebraElement) =
    GroupAlgebraElement(convert(CoordinateVector{T}, X.coordinates), X.product_matrix)

show{T}(io::IO, X::GroupAlgebraElement{T}) = print(io,
    "Element of Group Algebra over $(typeofelt(X)), of length $(length(X)):\n", X.coordinates)


function isequal{T, S}(X::GroupAlgebraElement{T}, Y::GroupAlgebraElement{S})
    if T != S
        warn("Comparing elements with different coefficients Rings!")
    end
    X.product_matrix == Y.product_matrix || return false
    X.coordinates == Y.coordinates || return false
    return true
end

(==)(X::GroupAlgebraElement, Y::GroupAlgebraElement) = isequal(X,Y)

function add{T<:CoordinateVector}(X::GroupAlgebraElement{T}, Y::GroupAlgebraElement{T})
    X.product_matrix == Y.product_matrix || throw(ArgumentError(
    "Elements don't seem to belong to the same Group Algebra!"))
    return GroupAlgebraElement(X.coordinates+Y.coordinates, X.product_matrix)
end

function add{T<:CoordinateVector, S<:CoordinateVector}(X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{S})
    warn("Adding elements with different base rings!")
    return GroupAlgebraElement(+(promote(X.coordinates, Y.coordinates)...),
    X.product_matrix)
end

(+)(X::GroupAlgebraElement, Y::GroupAlgebraElement) = add(X,Y)
(-)(X::GroupAlgebraElement) = GroupAlgebraElement(-X.coordinates, X.product_matrix)
(-)(X::GroupAlgebraElement, Y::GroupAlgebraElement) = add(X,-Y)

function group_star_multiplication{T<:CoordinateVector}(X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{T})
    X.product_matrix == Y.product_matrix || ArgumentError(
    "Elements don't seem to belong to the same Group Algebra!")
    result = zeros(X.coordinates)
    for (i,x) in enumerate(X.coordinates)
        if x != 0
            for (j,y) in enumerate(Y.coordinates)
                if y != 0
                    index = X.product_matrix[i,j]
                    if index == 0
                        throw(ArgumentError("The product don't seem to belong to the span of basis!"))
                    else
                        result[index]+= x*y
                    end
                end
            end
        end
    end
    return GroupAlgebraElement(result, X.product_matrix)
end

function group_star_multiplication{T<:CoordinateVector, S<:CoordinateVector}(
    X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{S})
    S == T || warn("Multiplying elements with different base rings!")
    return group_star_multiplication(promote(X,Y)...)
end

(*){T<:CoordinateVector, S<:CoordinateVector}(X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{S}) = group_star_multiplication(X,Y);

typeofelt{T<:Number}(X::AbstractVector{T}) = T
typeofelt{S<:CoordinateVector}(X::GroupAlgebraElement{S}) = typeofelt(X.coordinates)

function (*){T<:Number, S<:CoordinateVector}(a::T, X::GroupAlgebraElement{S})
    W = typeofelt(X)
    promote_type(T,W) == W || warn("Scalar and coordinates are in different rings! Promoting result to $(promote_type(T,W))")
    return GroupAlgebraElement(a*X.coordinates, X.product_matrix)
end

(*){T<:Number, S<:CoordinateVector}(X::GroupAlgebraElement{S}, a::T) = (*)(a, X)

function rational_division{T<:CoordinateVector, S<:Rational}(X::GroupAlgebraElement{T}, a::S)
    if typeofelt(X) <: Rational
        return GroupAlgebraElement(X.coordinates//a, X.product_matrix)
    else
        throw(ArgumentError("Rational division attempt on a GroupAlgebraElement of non-rational coefficients!"))
    end
end

(//)(X,a) = rational_division(X,a)
(//){S<:Integer}(X::GroupAlgebraElement, a::S) = //(X, Rational{S}(a))

length(X::GroupAlgebraElement) = length(X.coordinates)
size(X::GroupAlgebraElement) = size(X.coordinates)
norm(X::GroupAlgebraElement, p=2) = norm(X.coordinates, p)

end
