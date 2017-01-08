module GroupAlgebras

import Base: convert, show, isequal, ==
import Base: +, -, *, //
import Base: size, length, norm

export GroupAlgebraElement


immutable GroupAlgebraElement{T<:Number}
    coefficients::Vector{T}
    product_matrix::Array{Int,2}
    # basis::Array{Any,1}

    function GroupAlgebraElement(coefficients::Vector{T},
        product_matrix::Array{Int,2})

        size(product_matrix, 1) == size(product_matrix, 2) ||
            throw(ArgumentError("Product matrix has to be square"))
        new(coefficients, product_matrix)
    end
end

# GroupAlgebraElement(c,pm,b) = GroupAlgebraElement(c,pm)
GroupAlgebraElement{T}(c::Vector{T},pm) = GroupAlgebraElement{T}(c,pm)

convert{T<:Number}(::Type{T}, X::GroupAlgebraElement) =
    GroupAlgebraElement(convert(Vector{T}, X.coefficients), X.product_matrix)

show{T}(io::IO, X::GroupAlgebraElement{T}) = print(io,
    "Element of Group Algebra over $T of length $(length(X)):\n $(X.coefficients)")


function isequal{T, S}(X::GroupAlgebraElement{T}, Y::GroupAlgebraElement{S})
    if T != S
        warn("Comparing elements with different coefficients Rings!")
    end
    X.product_matrix == Y.product_matrix || return false
    X.coefficients == Y.coefficients || return false
    return true
end

(==)(X::GroupAlgebraElement, Y::GroupAlgebraElement) = isequal(X,Y)

function add{T<:Number}(X::GroupAlgebraElement{T}, Y::GroupAlgebraElement{T})
    X.product_matrix == Y.product_matrix || throw(ArgumentError(
    "Elements don't seem to belong to the same Group Algebra!"))
    return GroupAlgebraElement(X.coefficients+Y.coefficients, X.product_matrix)
end

function add{T<:Number, S<:Number}(X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{S})
    warn("Adding elements with different base rings!")
    return GroupAlgebraElement(+(promote(X.coefficients, Y.coefficients)...),
    X.product_matrix)
end

(+)(X::GroupAlgebraElement, Y::GroupAlgebraElement) = add(X,Y)
(-)(X::GroupAlgebraElement) = GroupAlgebraElement(-X.coefficients, X.product_matrix)
(-)(X::GroupAlgebraElement, Y::GroupAlgebraElement) = add(X,-Y)

function group_star_multiplication{T<:Number}(X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{T})
    X.product_matrix == Y.product_matrix || ArgumentError(
    "Elements don't seem to belong to the same Group Algebra!")
    result = zeros(X.coefficients)
    for (i,x) in enumerate(X.coefficients)
        if x != 0
            for (j,y) in enumerate(Y.coefficients)
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

function group_star_multiplication{T<:Number, S<:Number}(
    X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{S})
    S == T || warn("Multiplying elements with different base rings!")
    return group_star_multiplication(promote(X,Y)...)
end

(*){T<:Number, S<:Number}(X::GroupAlgebraElement{T},
    Y::GroupAlgebraElement{S}) = group_star_multiplication(X,Y);

(*){T<:Number}(a::T, X::GroupAlgebraElement{T}) = GroupAlgebraElement(
    a*X.coefficients, X.product_matrix)

function scalar_multiplication{T<:Number, S<:Number}(a::T,
    X::GroupAlgebraElement{S})
    promote_type(T,S) == S || warn("Scalar and coefficients are in different rings! Promoting result to $(promote_type(T,S))")
    return GroupAlgebraElement(a*X.coefficients, X.product_matrix)
end

(*){T<:Number}(a::T,X::GroupAlgebraElement) = scalar_multiplication(a, X)

//{T<:Rational, S<:Rational}(X::GroupAlgebraElement{T}, a::S) =
    GroupAlgebraElement(X.coefficients//a, X.product_matrix)

//{T<:Rational, S<:Integer}(X::GroupAlgebraElement{T}, a::S) =
    X//convert(T,a)

length(X::GroupAlgebraElement) = length(X.coefficients)
size(X::GroupAlgebraElement) = size(X.coefficients)
norm(X::GroupAlgebraElement, p=2) = norm(X.coefficients, p)

end
