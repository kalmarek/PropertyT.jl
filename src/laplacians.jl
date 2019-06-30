###############################################################################
#
#  Laplacians
#
###############################################################################

function spLaplacian(RG::GroupRing, S::AbstractVector{El}, T::Type=Float64) where El
    result = RG(T)
    id = (El <: AbstractAlgebra.NCRingElem ? one(RG.group) : RG.group())
    result[id] = T(length(S))
    for s in S
        result[s] -= one(T)
    end
    return result
end

function Laplacian(S::AbstractVector{REl}, halfradius) where REl<:AbstractAlgebra.NCRingElem
    R = parent(first(S))
    return Laplacian(S, one(R), halfradius)
end

function Laplacian(S::AbstractVector{E}, halfradius) where E<:AbstractAlgebra.GroupElem
    G = parent(first(S))
    return Laplacian(S, G(), halfradius)
end

function Laplacian(S, Id, halfradius)
    @info "Generating metric ball of radius" radius=2halfradius
    @time E_R, sizes = Groups.generate_balls(S, Id, radius=2halfradius)
    @info "Generated balls:" sizes

    @info "Creating product matrix..."
    rdict = GroupRings.reverse_dict(E_R)
    @time pm = GroupRings.create_pm(E_R, rdict, sizes[halfradius]; twisted=true)

    RG = GroupRing(parent(Id), E_R, rdict, pm)
    Δ = spLaplacian(RG, S)
    return Δ
end

function saveGRElem(fname::String, g::GroupRingElem)
    RG = parent(g)
    JLD.save(fname, "coeffs", g.coeffs, "pm", RG.pm, "G", RG.group)
end

function loadGRElem(fname::String, RG::GroupRing)
    coeffs = load(fname, "coeffs")
    return GroupRingElem(coeffs, RG)
end

function loadGRElem(fname::String, G::Union{Group, NCRing})
    pm = load(fname, "pm")
    RG = GroupRing(G, pm)
    return loadGRElem(fname, RG)
end

function loadGRElem(fname::String)
    pm, G = load(fname, "pm", "G")
    RG = GroupRing(G, pm)
    return loadGRElem(fname, RG)
end
