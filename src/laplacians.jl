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
    @info("Generating metric ball of radius $(2radius)...")
    @time E_R, sizes = Groups.generate_balls(S, Id, radius=2radius)
    @info("Generated balls of sizes $sizes.")

    @time pm = GroupRings.create_pm(E_R, GroupRings.reverse_dict(E_R), sizes[radius]; twisted=true)
    @info("Creating product matrix...")

    RG = GroupRing(parent(Id), E_R, pm)
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

function loadGRElem(fname::String, G::Group)
    pm = load(fname, "pm")
    RG = GroupRing(G, pm)
    return loadGRElem(fname, RG) 
end

function loadGRElem(fname::String)
    pm, G = load(fname, "pm", "G")
    RG = GroupRing(G, pm)
    return loadGRElem(fname, RG)
end
