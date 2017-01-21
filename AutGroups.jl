module AutGroups

using Groups
using Permutations

import Base: inv, ^
import Groups: IdSymbol, change_pow, GWord,  ==, hash, reduce!

export AutSymbol, AutWord, GWord
export rmul_AutSymbol, lmul_AutSymbol, flip_AutSymbol, symmetric_AutSymbol

immutable AutSymbol <: GSymbol
    gen::String
    pow::Int
    ex::Expr
end

(==)(s::AutSymbol, t::AutSymbol) = s.gen == t.gen && s.pow == t.pow
hash(s::AutSymbol, h::UInt) = hash(s.gen, hash(s.pow, hash(:AutSymbol, h)))

IdSymbol(::Type{AutSymbol}) = AutSymbol("(id)", 0, :(IdAutomorphism(N)))

function change_pow(s::AutSymbol, n::Int)

    if n == 0
        return one(s)
    end

    symbol = s.ex.args[1]
    if symbol == :ɛ
        return flip_AutSymbol(s.ex.args[2], pow=n)
    elseif symbol == :σ
        return symmetric_AutSymbol(s.ex.args[2], pow=n)
    elseif symbol == :ϱ
        return rmul_AutSymbol(s.ex.args[2], s.ex.args[3], pow=n)
    elseif symbol == :λ
        return lmul_AutSymbol(s.ex.args[2], s.ex.args[3], pow=n)
    elseif symbol == :IdAutomorphism
        return s
    else
        warn("Changing an unknown type of symbol! $s")
        return AutSymbol(s.gen, n, s.ex)
    end
end

inv(f::AutSymbol) = change_pow(f, -1*f.pow)
(^)(s::AutSymbol, n::Integer) = change_pow(s, s.pow*n)

function rmul_AutSymbol(i,j; pow::Int=1)
    gen = string('ϱ',Char(8320+i), Char(8320+j)...)
    return AutSymbol(gen, pow, :(ϱ($i,$j)))
end

function lmul_AutSymbol(i,j; pow::Int=1)
    gen = string('λ',Char(8320+i), Char(8320+j)...)
    return AutSymbol(gen, pow, :(λ($i,$j)))
end

function flip_AutSymbol(j; pow::Int=1)
    gen = string('ɛ', Char(8320 + j))
    return AutSymbol(gen, (2+ pow%2)%2, :(ɛ($j)))
end

function symmetric_AutSymbol(perm::Vector{Int}; pow::Int=1)
    # if perm == collect(1:length(perm))
    #     return one(AutSymbol)
    # end
    perm = Permutation(perm)
    ord = order(perm)
    pow = pow % ord
    perm = perm^pow
    gen = string('σ', [Char(8320 + i) for i in array(perm)]...)
    return AutSymbol(gen, 1, :(σ($(array(perm)))))
end

typealias AutWord GWord{AutSymbol}

end #end of module AutGroups
