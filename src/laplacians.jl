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

function computeλandP(sett::Settings{Naive},
    Δ::GroupRingElem, ws=nothing; solverlog=tempname()*".log")
    info("Creating SDP problem...")
    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, upper_bound=sett.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver)
    info(Base.repr(SDP_problem))

    @time λ, P, ws = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)

    return λ, P, ws
end

function computeλandP(sett::Settings{Symmetrize},
    Δ::GroupRingElem, ws=nothing; solverlog=tempname()*".log")
    pdir = prepath(sett)

    files_exist = exists(filename(pdir,:Uπs)) && exists(filename(pdir,:orbits)) && exists(filename(pdir,:preps))

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

    @time λ, P, ws = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)

    fname = filename(fullpath(sett), :P)
    save(joinpath(dirname(fname), "orig_"*basename(fname)), "origP", P)

    info("Reconstructing P...")
    @time recP = reconstruct(P, orbit_data)

    return λ, recP, ws
end

function saveλandP(name, λ, P, ws)
    save(filename(name, :λ), "λ", λ)
    save(filename(name, :P), "P", P)
    save(filename(name, :warm), "warmstart", ws)
end

function loadλandP(name::String)
    λ_fname = filename(name, :λ)
    P_fname = filename(name, :P)

    if exists(λ_fname) && exists(P_fname)
        info("Loading precomputed λ, P...")
        λ = load(λ_fname, "λ")
        P = load(P_fname, "P")
    else
        throw("You need to precompute $λ_fname and $P_fname to load it!")
    end
    return λ, P
end
