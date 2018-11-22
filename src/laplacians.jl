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

###############################################################################
#
#  λandP
#
###############################################################################

function computeλandP(sett::Settings{Naive}, Δ::GroupRingElem;
    solverlog=tempname()*".log")

    info("Creating SDP problem...")
    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, upper_bound=sett.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver)
    info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, (λ, P, ws) = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)
    save(filename(sett, :warmstart), "warmstart", ws)

    return λ, P
end

function computeλandP(sett::Settings{Symmetrize}, Δ::GroupRingElem;
    solverlog=tempname()*".log")

    if isfile(filename(sett, :OrbitData))
        orbit_data = load(filename(sett, :OrbitData), "OrbitData")
    else
        isdefined(parent(Δ), :basis) || throw("You need to define basis of Group Ring to compute orbit decomposition!")
        orbit_data = OrbitData(parent(Δ), sett.autS)
        save(filename(sett, :OrbitData), "OrbitData", orbit_data)
    end
    orbit_data = decimate(orbit_data)

    info("Creating SDP problem...")

    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, orbit_data, upper_bound=sett.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver)
    info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, (λ, Ps, ws) = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)
    save(filename(sett, :warmstart), "warmstart", ws, "Ps", Ps, "λ", λ)

    info("Reconstructing P...")
    @time P = reconstruct(Ps, orbit_data)

    return λ, P
end

function warmstart(sett::Settings)
    if sett.warmstart && isfile(filename(sett, :warmstart))
        ws = load(filename(sett, :warmstart), "warmstart")
    else
        ws = nothing
    end
    return ws
end
