using JuMP
using SCS

export Settings, OrbitData

immutable Settings{T<:AbstractMathProgSolver}
    name::String
    N::Int
    G::Group
    S::Vector
    autS::Group
    radius::Int
    solver::T
    upper_bound::Float64
    tol::Float64
    warmstart::Bool
end

prefix(s::Settings) = s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

struct OrbitData{T<:AbstractArray{Float64, 2}}
    orbits::Vector{Vector{Int}}
    Uπs::Vector{T}
    dims::Vector{Int}
end

function OrbitData(sett::Settings)
    info("Loading Uπs, dims, orbits...")
    Uπs = load(filename(prepath(sett), :Uπs), "Uπs")
    nzros = [i for i in 1:length(Uπs) if size(Uπs[i],2) !=0]
    Uπs = Uπs[nzros]

    Uπs = map(x -> sparsify!(x, sett.tol/100, verbose=true), Uπs)
    #dimensions of the corresponding πs:
    dims = load(filename(prepath(sett), :Uπs), "dims")[nzros]

    orbits = load(filename(prepath(sett), :orbits), "orbits");

    return OrbitData(orbits, Uπs, dims)
end

include("OrbitDecomposition.jl")

dens(M::SparseMatrixCSC) = nnz(M)/length(M)
dens(M::AbstractArray) = countnz(M)/length(M)

function sparsify!{Tv,Ti}(M::SparseMatrixCSC{Tv,Ti}, eps=eps(Tv); verbose=false)

    densM = dens(M)
    for i in eachindex(M.nzval)
        if abs(M.nzval[i]) < eps
            M.nzval[i] = zero(Tv)
        end
    end
    dropzeros!(M)

    if verbose
        info("Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M), 20), " ($(nnz(M)) non-zeros)")
    end

    return M
end

function sparsify!{T}(M::AbstractArray{T}, eps=eps(T); verbose=false)
    densM = dens(M)
    if verbose
        info("Sparsifying $(size(M))-matrix... ")
    end

    for n in eachindex(M)
        if abs(M[n]) < eps
            M[n] = zero(T)
        end
    end

    if verbose
        info("$(rpad(densM, 20)) → $(rpad(dens(M),20))), ($(countnz(M)) non-zeros)")
    end

    return sparse(M)
end

sparsify{T}(U::AbstractArray{T}, tol=eps(T); verbose=false) = sparsify!(deepcopy(U), tol, verbose=verbose)

function constrLHS(m::JuMP.Model, cnstr, Us, Ust, dims, vars, eps=100*eps(1.0))
    M = [PropertyT.sparsify!(dims[π].*Ust[π]*cnstr*Us[π], eps) for π in 1:endof(Us)]
    return @expression(m, sum(vecdot(M[π], vars[π]) for π in 1:endof(Us)))
end

function addconstraints!(m::JuMP.Model, X::GroupRingElem, orderunit::GroupRingElem, λ::JuMP.Variable, P, data::OrbitData)

    orderunit_orb = orbit_spvector(orderunit.coeffs, data.orbits)
    X_orb = orbit_spvector(X.coeffs, data.orbits)
    Ust = [U' for U in data.Uπs]
    n = size(parent(X).pm, 1)

    for t in 1:length(X_orb)
        x, u = X_orb[t], orderunit_orb[t]
        cnstrs = [constraint(parent(X).pm, o) for o in data.orbits[t]]
        lhs = constrLHS(m, orbit_constraint(cnstrs,n), data.Uπs, Ust, data.dims, P)

        JuMP.@constraint(m, lhs == x - λ*u)
    end
end

function init_model(m, sizes)
    P = Vector{Array{JuMP.Variable,2}}(length(sizes))

    for (k,s) in enumerate(sizes)
        P[k] = JuMP.@variable(m, [i=1:s, j=1:s])
        JuMP.@SDconstraint(m, P[k] >= 0.0)
    end

    return P
end

function SOS_problem(X::GroupRingElem, orderunit::GroupRingElem, data::OrbitData; upper_bound=Inf)
    m = JuMP.Model();
    P = init_model(m, size.(data.Uπs,2))

    λ = JuMP.@variable(m, λ)
    if upper_bound < Inf
        JuMP.@constraint(m, λ <= upper_bound)
    end

    info("Adding $(length(data.orbits)) constraints... ")

    @time addconstraints!(m, X, orderunit, λ, P, data)

    JuMP.@objective(m, Max, λ)
    return m, λ, P
end

function computeλandP(Δ::GroupRingElem, sett::Settings, ws=nothing; solverlog=tempname()*".log")
    @time orbit_data = OrbitData(sett);
    info("Creating SDP problem...")

    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, orbit_data, upper_bound=sett.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver)
    info(Base.repr(SDP_problem))

    @time λ, P, ws = solve_SDP(SDP_problem, varλ, varP, ws, solverlog=solverlog)

    fname = filename(fullpath(sett), :P)
    save(joinpath(dirname(fname), "orig_"*basename(fname)), "origP", P)

    info("Reconstructing P...")
    preps = load_preps(filename(prepath(sett), :preps), sett.autS)
    @time recP = reconstruct_sol(preps, orbit_data.Uπs, P, orbit_data.dims)

    return λ, recP, ws
end

function load_preps(fname::String, G::Group)
    lded_preps = load(fname, "perms_d")
    permG = PermutationGroup(length(first(lded_preps)))
    @assert length(lded_preps) == order(G)
    return Dict(k=>permG(v) for (k,v) in zip(elements(G), lded_preps))
end

function save_preps(fname::String, preps)
    autS = parent(first(keys(preps)))
    save(fname, "perms_d", [preps[elt].d for elt in elements(autS)])
end

function check_property_T(sett::Settings)

    ex(s::Symbol) = exists(filename(prepath(sett), s))

    if ex(:pm) && ex(:Δ)
        # cached
        Δ = loadLaplacian(prepath(sett), parent(sett.S[1]))
    else
        # compute
        Δ = computeLaplacian(sett.S, sett.radius)
        save(filename(prepath(sett), :pm), "pm", parent(Δ).pm)
        save(filename(prepath(sett), :Δ), "Δ", Δ.coeffs)
    end

    files_exist = ex.([:Uπs, :orbits, :preps])

    if !all(files_exist)
        compute_orbit_data(prepath(sett), parent(Δ), sett.autS)
    end

    files_exist = exists(filename(fullpath(sett), :λ)) &&
        exists(filename(fullpath(sett), :P))

    if !sett.warmstart && files_exist
        λ, P = loadλandP(fullpath(sett))
    else
        warmfile = filename(fullpath(sett), :warm)
        if sett.warmstart && exists(warmfile)
            ws = load(warmfile, "warmstart")
        else
            ws = nothing
        end
        λ, P, ws = computeλandP(Δ, sett, ws,
            solverlog=filename(fullpath(sett), :solverlog))
        saveλandP(fullpath(sett), λ, P, ws)

        if λ < 0
            warn("Solver did not produce a valid solution!")
        end
    end

    info("λ = $λ")
    info("sum(P) = $(sum(P))")
    info("maximum(P) = $(maximum(P))")
    info("minimum(P) = $(minimum(P))")

    isapprox(eigvals(P), abs.(eigvals(P)), atol=sett.tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")

    return interpret_results(sett.name, Δ, sett.radius, length(sett.S), λ, P)
end
