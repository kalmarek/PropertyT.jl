###############################################################################
#
#  Laplacians
#
###############################################################################

function spLaplacian(RG::GroupRing, S::AbstractVector, T::Type=Float64)
    result = RG(T)
    result[one(RG.group)] = T(length(S))
    for s in S
        result[s] -= one(T)
    end
    return result
end

function Laplacian(S::AbstractVector{REl}, halfradius) where REl<:Union{NCRingElem, GroupElem}
    G = parent(first(S))
    @info "Generating metric ball of radius" radius=2halfradius
    @time E_R, sizes = Groups.wlmetric_ball(S, radius=2halfradius)
    @info "Generated balls:" sizes

    @info "Creating product matrix..."
    rdict = GroupRings.reverse_dict(E_R)
    @time pm = GroupRings.create_pm(E_R, rdict, sizes[halfradius]; twisted=true)

    RG = GroupRing(G, E_R, rdict, pm)
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
