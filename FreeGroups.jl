module FreeGroups

using Groups

import Base: inv, convert

export FGSymbol, IDSymbol

immutable FGSymbol <: GSymbol
    gen::String
    pow::Int
end

IDSymbol(::Type{FGSymbol}) = FGSymbol("(id)", 0)
FGSymbol(x::String) = FGSymbol(x,1)

inv(s::FGSymbol) = FGSymbol(s.gen, -s.pow)
convert(::Type{FGSymbol}, x::String) = FGSymbol(x)
change_pow(s::FGSymbol, n::Int) = reduce(FGSymbol(s.gen, n))

typealias FGWord GWord{FGSymbol}

FGWord(s::FGSymbol) = FGWord([s])

end #end of module FreeGroups
