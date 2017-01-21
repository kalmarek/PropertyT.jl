module AutGroups

using Groups
using Permutations

import Base: inv
import Groups: IdSymbol, change_pow, GWord,  ==, hash, reduce!

export IDSymbol, AutSymbol, AutWord
export rmul_AutSymbol, lmul_AutSymbol, flip_AutSymbol, symmetric_AutSymbol

immutable AutSymbol <: GSymbol
    gen::String
    pow::Int
    ex::Expr
end

(==)(s::AutSymbol, t::AutSymbol) = s.gen == t.gen && s.pow == t.pow
hash(s::AutSymbol, h::UInt) = hash(s.gen, hash(s.pow, hash(:AutSymbol, h)))
IDSymbol(::Type{AutSymbol}) = AutSymbol("(id)", 0, :(IDAutomorphism(N)))

change_pow(s::AutSymbol, n::Int) = reduce(AutSymbol(s.gen, n, s.ex))

function inv(f::AutSymbol)
    symbol = f.ex.args[1]
    if symbol == :ɛ
        return change_pow(f, f.pow % 2)
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

function rmul_AutSymbol(i,j, pow::Int=1)
    gen = string('ϱ',Char(8320+i), Char(8320+j)...)
    return AutSymbol(gen, pow, :(ϱ($i,$j)))
end

function lmul_AutSymbol(i,j, pow::Int=1)
    gen = string('λ',Char(8320+i), Char(8320+j)...)
    return AutSymbol(gen, pow, :(λ($i,$j)))
end

function flip_AutSymbol(j, pow::Int=1)
    gen = string('ɛ', Char(8320 + j))
    return AutSymbol(gen, pow%2, :(ɛ($j)))
end

function symmetric_AutSymbol(perm::Vector{Int}, pow::Int=1)
    perm = Permutation(perm)
    ord = order(perm)
    pow = pow % ord
    perm = perm^pow
    gen = string('σ', [Char(8320 + i) for i in array(perm)]...)
    return AutSymbol(gen, 1, :(σ($(array(perm)))))
end

typealias AutWord GWord{AutSymbol}

end #end of module AutGroups
