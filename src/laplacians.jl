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

function spLaplacian(RG::GroupRing, S::Vector{REl}, T::Type=Float64) where {REl<:AbstractAlgebra.ModuleElem}
    result = RG(T)
    result[one(RG.group)] = T(length(S))
    for s in S
        result[s] -= one(T)
    end
    return result
end

function Laplacian(S::Vector{E}, radius) where E<:AbstractAlgebra.ModuleElem
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

    @info("Creating product matrix...")
    rdict = GroupRings.reverse_dict(E_R)
    @time pm = GroupRings.create_pm(E_R, rdict, sizes[radius]; twisted=true)

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
