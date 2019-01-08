###############################################################################
#
#  Laplacians
#
###############################################################################

function spLaplacian(RG::GroupRing, S, T::Type=Float64)
    result = RG(T)
    result[RG.group()] = T(length(S))
    for s in S
        result[s] -= one(T)
    end
    return result
end

function spLaplacian(RG::GroupRing{R}, S, T::Type=Float64) where {R<:Ring}
    result = RG(T)
    result[one(RG.group)] = T(length(S))
    for s in S
        result[s] -= one(T)
    end
    return result
end

function Laplacian(S::Vector{E}, radius) where E<:AbstractAlgebra.RingElem
    R = parent(first(S))
    return Laplacian(S, one(R), radius)
end

function Laplacian(S::Vector{E}, radius) where E<:AbstractAlgebra.GroupElem
    G = parent(first(S))
    return Laplacian(S, G(), radius)
end

function Laplacian(S, Id, radius)
    info("Generating metric ball of radius $(2radius)...")
    @time E_R, sizes = Groups.generate_balls(S, Id, radius=2radius)
    info("Generated balls of sizes $sizes.")

    info("Creating product matrix...")
    @time pm = GroupRings.create_pm(E_R, GroupRings.reverse_dict(E_R), sizes[radius]; twisted=true)

    RG = GroupRing(parent(Id), E_R, pm)
    Δ = spLaplacian(RG, S)
    return Δ
end

function saveGRElem(filename::String, g::GroupRingElem)
    RG = parent(g)
    JLD.save(filename, "coeffs", g.coeffs, "pm", RG.pm, "G", RG.group)
end

function loadGRElem(fname::String, G::Group)
    if isfile(fname)
        info("Loading precomputed Δ...")
        coeffs, pm = load(fname, "coeffs", "pm")
        RG = GroupRing(G, pm)
        Δ = GroupRingElem(coeffs, RG)
    else
        throw(ErrorException("You need to precompute $fname first!"))
    end
    return Δ
end
