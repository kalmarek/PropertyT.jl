module FreeGroups

export GSymbol, FGSymbol, Word, GWord, FGWord, FGAutomorphism

import Base: length, ==, show, convert
import Base: *, ^, convert
import Base: one, inv, reduce, push!, unshift!

abstract GSymbol

immutable FGSymbol <: GSymbol
    gen::String
    pow::Int
end

FGSymbol(x::String) = FGSymbol(x,1)

function show(io::IO, s::GSymbol)
    if s.pow == 1
        print(io, (s.gen))
    elseif s.pow == 0
        print(io, "(id)")
    else
        print(io, (s.gen)*"^$(s.pow)")
    end
end

(==)(s::GSymbol, t::GSymbol) = s.gen == t.gen && s.pow == t.pow
length(s::GSymbol) = (s.pow == 0 ? 0 : 1)

one(s::FGSymbol) = FGSymbol(s.gen, 0)
inv(s::FGSymbol) = FGSymbol(s.gen, -s.pow)

convert(::Type{FGSymbol}, x::String) = FGSymbol(x)

reduce!(s::GSymbol) = s
change_pow(s::FGSymbol, n::Int) = FGSymbol(s.gen, n)

(^)(x::FGSymbol, n::Integer) = FGSymbol(x.gen, x.pow*n)

function (*)(s::GSymbol, t::GSymbol)
    W = promote_type(typeof(s), typeof(t))
    return GWord{W}([s])*t
end


abstract Word

immutable GWord{T<:GSymbol} <: Word
    symbols::Vector{T}
end

typealias FGWord GWord{FGSymbol}

GWord{T<:GSymbol}(s::T) = GWord{T}([s])
FGWord(s::FGSymbol) = FGWord([s])
# FGWord() = FGWord(Vector{FGSymbol}())

function length(W::GWord)
    return sum([abs(s.pow) for s in W.symbols])
end

one{T}(::Type{GWord{T}}) = GWord(Vector{T}())
one{T}(w::GWord{T}) = one(GWord{T})

function inv{T}(W::GWord{T})
    if length(W) == 0
        return W
    else
        return prod(reverse([inv(s) for s in W.symbols]))
    end
end

function reduce!(W::GWord{FGSymbol})
    if length(W) < 2
        deleteat!(W.symbols, find(x -> x.pow == 0, W.symbols))
        return W
    end

    reduced = false
    while !reduced
        reduced = true
        for i in 1:length(W.symbols) - 1
            if W.symbols[i].gen == W.symbols[i+1].gen
                reduced = false
                p1 = W.symbols[i].pow
                p2 = W.symbols[i+1].pow
                W.symbols[i+1] = change_pow(W.symbols[i], p1 + p2)
                W.symbols[i] = one(W.symbols[i])
            end
        end
        deleteat!(W.symbols, find(x -> x.pow == 0, W.symbols))
    end
    return W
end

reduce(W::GWord) = reduce!(deepcopy(W))

(==)(W::GWord{FGSymbol}, Z::GWord{FGSymbol}) = reduce!(W).symbols == reduce!(Z).symbols

function show(io::IO, W::GWord)
    if length(W) == 0
        print(io, "(id)")
    else
        join(io, [string(s) for s in W.symbols], "*")
    end
end

push!(W::GWord, x) = push!(W.symbols, x...)
unshift!(W::GWord, x) = unshift!(W.symbols, x...)

function r_multiply!(W::GWord, x; reduced::Bool=true)
    if length(x) > 0
        push!(W, x)
    end
    if reduced
        reduce!(W)
    end
    return W
end

function l_multiply!(W::GWord, x; reduced::Bool=true)
    if length(x) > 0
        unshift!(W, reverse(x))
    end
    if reduced
        reduce!(W)
    end
    return W
end

r_multiply(W::GWord, x; reduced::Bool=true) =
    r_multiply!(deepcopy(W),x, reduced=reduced)
l_multiply(W::GWord, x; reduced::Bool=true) =
    l_multiply!(deepcopy(W),x, reduced=reduced)

(*){T}(W::GWord{T}, Z::GWord{T}) = FreeGroups.r_multiply(W, Z.symbols)
(*)(W::GWord, s::GSymbol) = W*GWord(s)
(*)(s::GSymbol, W::GWord) = GWord(s)*W

function power_by_squaring{T}(x::GWord{T}, p::Integer)
    if p < 0
        return power_by_squaring(inv(x), -p)
    elseif p == 0
        return one(x)
    elseif p == 1
        return deepcopy(x)
    elseif p == 2
        return x*x
    end
    t = trailing_zeros(p) + 1
    p >>= t
    while (t -= 1) > 0
        x *= x
    end
    y = x
    while p > 0
        t = trailing_zeros(p) + 1
        p >>= t
        while (t -= 1) >= 0
            x *= x
        end
        y *= x
    end
    return reduce!(y)
end

(^)(x::GWord, n::Integer) = power_by_squaring(x,n)



type FGAutomorphism{T<:GSymbol}
    domain::Vector{T}
    image::Vector{GWord{T}}
    map::Function

    function FGAutomorphism{T}(domain::Vector{T}, image::Vector{GWord{T}}, map::Function)
        length(domain) == length(unique(domain)) ||
            throw(ArgumentError("The elements of $domain are not unique"))
        length(domain) == length(image) ||
            throw(ArgumentError("Dimensions of image and domain must match"))
#         Set(vcat([[s.gen for s in reduce!(x).symbols]
#             for x in image]...)) == Set(s.gen for s in domain) ||
#             throw(ArgumentError("Are You sure that $image defines an automorphism??"))
        new(domain, image, map)
    end
end

function show(io::IO, X::FGAutomorphism)
    title = "Endomorphism of Free Group on $(length(X.domain)) generators, sending"
    map = ["$x ⟶ $y" for (x,y) in zip(X.domain, X.image)]
    join(io, vcat(title,map), "\n")
end

(==)(f::FGAutomorphism, g::FGAutomorphism) =
    f.domain == g.domain && f.image == g.image

function aut_func_from_table(table::Vector{Tuple{Int,Int}}, GroupIdentity=one(FGWord))
    if length(table) == 0
        # warn("The map is not an automorphism")
        nothing
    end
    return v->reduce(*,GroupIdentity, v[idx]^power for (idx, power) in table)
end

function aut_func_from_word(domain, w::GWord)
    table = Vector{Tuple{Int, Int}}()
    for s in w.symbols
        pair = (findfirst([x.gen for x in domain], s.gen), s.pow)
        push!(table, pair)
    end
    return aut_func_from_table(table)
end

function FGMap(domain::Vector{FGSymbol}, image::Vector{GWord})

    function_vector = Vector{Function}()

    for word in image
        push!(function_vector, aut_func_from_word(domain, word))
    end

    return v -> Vector{FGWord}([f(v) for f in function_vector])
end

FGAutomorphism(domain::Vector{FGSymbol}, image::Vector{GWord}) =
    FGAutomorphism(domain, image, FGMap(domain, image))

FGAutomorphism(domain::Vector{FGSymbol}, image::Vector{FGSymbol}) =
    FGAutomorphism(domain, Vector{GWord}(image))

function FGAutomorphism(domain::Vector, image::Vector)
    FGAutomorphism(Vector{FGSymbol}(domain), Vector{GWord}(image))
end

function FGAutomorphism(domain, image)
    FGAutomorphism([domain...], [image...])
end

"""Computes the composition g∘f of two morphisms"""
function compose(f::FGAutomorphism, g::FGAutomorphism)
    if length(f.image) != length(g.domain)
        throw(ArgumentError("Cannot compose $f and $g"))
    else
        h(v) = g.map(f.map(v))
        return FGAutomorphism(f.domain, h(f.domain), h)
    end
end

(*)(f::FGAutomorphism, g::FGAutomorphism) = compose(f,g)

end
