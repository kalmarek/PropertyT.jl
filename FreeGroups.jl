module FreeGroups

export FGSymbol, FGWord, FGAutomorphism

import Base: length, ==, show, convert

immutable FGSymbol
    gen::String
    pow::Int
end

(==)(s::FGSymbol, t::FGSymbol) = s.gen == t.gen && s.pow == t.pow

immutable FGWord
    symbols::Vector{FGSymbol}
end

length(s::FGSymbol) = (s.pow == 0 ? 0 : 1)

length(W::FGWord) = length(W.symbols)

function show(io::IO, s::FGSymbol)
    if s.pow == 1
        print(io, (s.gen))
    elseif s.pow == 0
        print(io, "(id)")
    else
        print(io, (s.gen)*"^$(s.pow)")
    end
end

FGSymbol(x::String) = FGSymbol(x,1)
FGWord() = FGWord(Vector{FGSymbol}())
FGWord(s::FGSymbol) = FGWord([s])

convert(::Type{FGWord}, s::FGSymbol) = FGWord(s)


import Base: one, inv, reduce, push!, unshift!

one(s::FGSymbol) = FGSymbol(s.gen, 0)
one(::Type{FGWord}) = FGWord()
one(w::FGWord) = FGWord()

inv(s::FGSymbol) = FGSymbol(s.gen, -s.pow)
inv(W::FGWord) = FGWord(reverse([inv(s) for s in W.symbols]))

reduce!(s::FGSymbol) = s

function reduce!(W::FGWord)
    for i in 1:length(W)-1
        if W.symbols[i].gen == W.symbols[i+1].gen
            p1 = W.symbols[i].pow
            p2 = W.symbols[i+1].pow
            W.symbols[i+1] = FGSymbol(W.symbols[i].gen, p1 + p2)
            W.symbols[i] = one(W.symbols[i])
        end
    end
    deleteat!(W.symbols, find(x -> x.pow == 0, W.symbols))
    return W
end

reduce(W::FGWord) = reduce!(deepcopy(W))

(==)(W::FGWord, Z::FGWord) = reduce(W).symbols == reduce(Z).symbols

function show(io::IO, W::FGWord)
    if length(W) == 0
        print(io, "(id)")
    else
        join(io, [string(s) for s in W.symbols], "*")
    end
end;

push!(W::FGWord, x...) = push!(W.symbols, x...)
unshift!(W::FGWord, x...) = unshift!(W.symbols, reverse(x)...)

function r_multiply!(W::FGWord, x...; reduced::Bool=true)
    if length(x) > 0
        push!(W, x...)
    end
    if reduced
        reduce!(W)
    end
    return W
end

function l_multiply!(W::FGWord, x...; reduced::Bool=true)
    if length(x) > 0
        unshift!(W, x...)
    end
    if reduced
        reduce!(W)
    end
    return W
end

r_multiply(W::FGWord, x...; reduced::Bool=true) =
    r_multiply!(deepcopy(W),x..., reduced=reduced)
l_multiply(W::FGWord, x...; reduced::Bool=true) =
    l_multiply!(deepcopy(W),x..., reduced=reduced)

import Base: *, ^

(*)(W::FGWord, Z::FGWord) = r_multiply(W, Z.symbols...)
(*)(s::FGSymbol, t::FGSymbol) = FGWord(s)*FGWord(t)
(*)(W::FGWord, s::FGSymbol) = W*FGWord(s)
(*)(s::FGSymbol, W::FGWord) = FGWord(s)*W

(^)(x::FGSymbol, n::Integer) = FGSymbol(x.gen, x.pow*n)

function power_by_squaring(x::FGWord, p::Integer)
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

(^)(x::FGWord, n::Integer) = power_by_squaring(x,n)

type FGAutomorphism
    domain::Vector{FGSymbol}
    image::Vector{FGWord}
    map::Function

    function FGAutomorphism(domain::Vector{FGSymbol}, image::Vector{FGWord}, map::Function)
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

function aut_func_from_word(domain, w::FGWord)
    table = Vector{Tuple{Int, Int}}()
    for s in w.symbols
        pair = (findfirst([x.gen for x in domain], s.gen), s.pow)
        push!(table, pair)
    end
    return aut_func_from_table(table)
end

function FGMap(domain::Vector{FGSymbol}, image::Vector{FGWord})

    function_vector = Vector{Function}()

    for word in image
        push!(function_vector, aut_func_from_word(domain, word))
    end

    return v -> Vector{FGWord}([f(v) for f in function_vector])
end

FGAutomorphism(domain::Vector{FGSymbol}, image::Vector{FGWord}) =
    FGAutomorphism(domain, image, FGMap(domain, image))

FGAutomorphism(domain::Vector{FGSymbol}, image::Vector{FGSymbol}) =
    FGAutomorphism(domain, Vector{FGWord}(image))

function FGAutomorphism(domain::Vector, image::Vector)
    FGAutomorphism(Vector{FGSymbol}(domain), Vector{FGWord}(image))
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
