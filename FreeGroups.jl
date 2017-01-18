module FreeGroups

export GSymbol, AutSymbol, Word, GWord, FGWord, AutWord, FGAutomorphism

import Base: length, ==, show, convert
import Base: *, ^, convert
import Base: one, inv, reduce, push!, unshift!

abstract GSymbol

immutable FGSymbol <: GSymbol
    gen::String
    pow::Int
end

immutable AutSymbol <: GSymbol
    gen::String
    pow::Int
    ex::Expr
end

IDSymbol(::Type{FGSymbol}) = FGSymbol("(id)", 0)
IDSymbol(::Type{AutSymbol}) = AutSymbol("(id)", 0, :(IDAutomorphism(N)))
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

one{T<:GSymbol}(::Type{T}) = IDSymbol(T)
one(s::GSymbol) = one(typeof(s))
inv(s::FGSymbol) = FGSymbol(s.gen, -s.pow)

convert(::Type{FGSymbol}, x::String) = FGSymbol(x)

reduce(s::GSymbol) = (s.pow == 0 ? one(s) : s)
change_pow(s::FGSymbol, n::Int) = reduce(FGSymbol(s.gen, n))
change_pow(s::AutSymbol, n::Int) = reduce(AutSymbol(s.gen, n, s.ex))

(^)(s::GSymbol, n::Integer) = change_pow(s, s.pow*n)


function inv(f::AutSymbol)
    symbol = f.ex.args[1]
    if symbol == :ɛ
        return FreeGroups.change_pow(f, f.pow % 2)
    elseif symbol == :σ
        perm = invperm(f.ex.args[2])
        gen = string('σ', [Char(8320 + i) for i in perm]...)
        return AutSymbol(gen, f.pow, :(σ($perm)))
    elseif symbol == :(ϱ) || symbol == :λ
        return AutSymbol(f.gen, -f.pow, f.ex)
    elseif symbol == :IDAutomorphism
        return f
    else
        throw(ArgumentError("Don't know how to invert $f (of type $symbol)"))
    end
end

function (*){T<:GSymbol}(s::T, t::T)
    return GWord{T}([s])*t
end


abstract Word

immutable GWord{T<:GSymbol} <: Word
    symbols::Vector{T}
end

typealias FGWord GWord{FGSymbol}
typealias AutWord GWord{AutSymbol}

GWord{T<:GSymbol}(s::T) = GWord{T}([s])
FGWord(s::FGSymbol) = FGWord([s])

IDWord{T<:GSymbol}(::Type{T}) = GWord(one(T))
IDWord{T<:GSymbol}(W::GWord{T}) = IDWord(T)

function length(W::GWord)
    return sum([abs(s.pow) for s in W.symbols])
end

one{T}(::Type{GWord{T}}) = IDWord(T)
one{T}(w::GWord{T}) = one(GWord{T})

function inv{T}(W::GWord{T})
    if length(W) == 0
        return W
    else
        return prod(reverse([inv(s) for s in W.symbols]))
    end
end

function free_group_reduction!(W::GWord)
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
    return reduced
end

function reduce!{T}(W::GWord{T}, reduce_func::Function=free_group_reduction!)
    if length(W) < 2
        deleteat!(W.symbols, find(x -> x.pow == 0, W.symbols))
        return W
    end

    reduced = false
    while !reduced
        reduced = reduce_func(W)
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
