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
    logger
end

prefix(s::Settings) = s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

immutable OrbitData{T<:AbstractArray{Float64, 2}, LapType <:AbstractVector{Float64}}
    name::String
    Us::Vector{T}
    Ps::Vector{Array{JuMP.Variable,2}}
    cnstr::Vector{SparseMatrixCSC{Float64, Int}}
    laplacian::LapType
    laplacianSq::LapType
    dims::Vector{Int}
end

function OrbitData(sett::Settings)
    splap = load(filename(prepath(sett), :Δ), "Δ");
    pm = load(filename(prepath(sett), :pm), "pm");
    cnstr = PropertyT.constraints(pm);
    splap² = similar(splap)
    splap² = GroupRings.mul!(splap², splap, splap, pm);

    Uπs = load(filename(prepath(sett), :Uπs), "Uπs")
    nzros = [i for i in 1:length(Uπs) if size(Uπs[i],2) !=0]
    Uπs = Uπs[nzros]
    Uπs = sparsify!.(Uπs, sett.tol, check=true, verbose=true)

    #dimensions of the corresponding πs:
    dims = load(filename(prepath(sett), :Uπs), "dims")[nzros]

    m, P = init_model(size(Uπs,1), [size(U,2) for U in Uπs]);

    orbits = load(filename(prepath(sett), :orb), "orbits");
    n = size(Uπs[1],1)
    orb_spcnstrm = [orbit_constraint(cnstr[collect(orb)], n) for orb in orbits]
    orb_splap = orbit_spvector(splap, orbits)
    orb_splap² = orbit_spvector(splap², orbits)

    orbData = OrbitData(fullpath(sett), Uπs, P, orb_spcnstrm, orb_splap, orb_splap², dims);

    # orbData = OrbitData(name, Uπs, P, orb_spcnstrm, splap, splap², dims);

    return m, orbData
end

include("OrbitDecomposition.jl")

dens(M::SparseMatrixCSC) = length(M.nzval)/length(M)
dens(M::AbstractArray) = length(findn(M)[1])/length(M)

function sparsify!{Tv,Ti}(M::SparseMatrixCSC{Tv,Ti}, eps=eps(Tv); verbose=false)
    n = nnz(M)

    densM = dens(M)
    for i in eachindex(M.nzval)
        if abs(M.nzval[i]) < eps
            M.nzval[i] = zero(Tv)
        end
    end
    dropzeros!(M)
    m = nnz(M)

    if verbose
        info(LOGGER, "Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M), 20))
    end

    return M
end

function sparsify!{T}(M::AbstractArray{T}, eps=eps(T); check=false, verbose=false)
    densM = dens(M)
    rankM = rank(M)
    M[abs.(M) .< eps] .= zero(T)

    if check && rankM != rank(M)
        warn(LOGGER, "Sparsification decreased the rank!")
    end

    if verbose
        info(LOGGER, "Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M),20))
    end

    return sparse(M)
end

sparsify{T}(U::AbstractArray{T}, tol=eps(T); check=true, verbose=false) = sparsify!(deepcopy(U), tol, check=check, verbose=verbose)

function transform(U::AbstractArray, V::AbstractArray; sparse=true)
    if sparse
        return sparsify!(U'*V*U)
    else
        return U'*V*U
    end
end

A(data::OrbitData, π, t) = data.dims[π].*transform(data.Us[π], data.cnstr[t])

function constrLHS(m::JuMP.Model, data::OrbitData, t)
    l = endof(data.Us)
    lhs = @expression(m, sum(vecdot(A(data, π, t), data.Ps[π]) for π in 1:l))
    return lhs
end

function constrLHS(m::JuMP.Model, cnstr, Us, Ust, dims, vars, eps=100*eps(1.0))
    M = [PropertyT.sparsify!(dims[π].*Ust[π]*cnstr*Us[π], eps) for π in 1:endof(Us)]
    return @expression(m, sum(vecdot(M[π], vars[π]) for π in 1:endof(Us)))
end

function addconstraints!(m::JuMP.Model, data::OrbitData, l::Int=length(data.laplacian); var::Symbol=:λ)
    λ = m[var]
    Ust = [U' for U in data.Us]
    idx = [π for π in 1:endof(data.Us) if size(data.Us[π],2) != 0]

    for t in 1:l
        if t % 100 == 0
            print(t, ", ")
        end
        #   lhs = constrLHS(m, data, t)
        lhs = constrLHS(m, data.cnstr[t], data.Us[idx], Ust[idx], data.dims[idx], data.Ps[idx])

        d, d² = data.laplacian[t], data.laplacianSq[t]
        # if lhs == zero(lhs)
        #    if d == 0 && d² == 0
        #        info("Detected empty constraint")
        #        continue
        #    else
        #        warn("Adding unsatisfiable constraint!")
        #    end
        # end
        JuMP.@constraint(m, lhs == d² - λ*d)
    end
    println("")
end

function init_model(n, sizes)
    m = JuMP.Model();
    P = Vector{Array{JuMP.Variable,2}}(n)

    for (k,s) in enumerate(sizes)
        P[k] = JuMP.@variable(m, [i=1:s, j=1:s])
        JuMP.@SDconstraint(m, P[k] >= 0.0)
    end

    JuMP.@variable(m, λ >= 0.0)
    JuMP.@objective(m, Max, λ)
    return m, P
end

function create_SDP_problem(sett::Settings)
    info(sett.logger, "Loading orbit data....")
    @logtime sett.logger SDP_problem, orb_data = OrbitData(sett);

    if sett.upper_bound < Inf
        λ = JuMP.getvariable(SDP_problem, :λ)
        JuMP.@constraint(SDP_problem, λ <= sett.upper_bound)
    end

    t = length(orb_data.laplacian)
    info(sett.logger, "Adding $t constraints ... ")
    @logtime sett.logger addconstraints!(SDP_problem, orb_data)

    return SDP_problem, orb_data
end

function λandP(m::JuMP.Model, data::OrbitData, warmstart=true)
    varλ = m[:λ]
    varP = data.Ps
    λ, Ps = PropertyT.λandP(data.name, m, varλ, varP, warmstart)
    return λ, Ps
end

function λandP(m::JuMP.Model, data::OrbitData, sett::Settings)
    info(sett.logger, "Solving SDP problem...")
    λ, Ps = λandP(m, data, sett.warmstart)

    info(sett.logger, "Reconstructing P...")

    preps = load_preps(filename(prepath(sett), :preps), sett.autS)

    @logtime sett.logger recP = reconstruct_sol(preps, data.Us, Ps, data.dims)

    fname = filename(fullpath(sett), :P)
    save(fname, "origP", Ps, "P", recP)
    return λ, recP
end

function load_preps(fname::String, G::Nemo.Group)
    lded_preps = load(fname, "perms_d")
    permG = PermutationGroup(length(first(lded_preps)))
    @assert length(lded_preps) == order(G)
    return Dict(k=>permG(v) for (k,v) in zip(elements(G), lded_preps))
end

function save_preps(fname::String, preps)
    autS = parent(first(keys(preps)))
    JLD.save(fname, "perms_d", [preps[elt].d for elt in elements(autS)])
end

function check_property_T(sett::Settings)

    ex(s) = exists(filename(prepath(sett), s))

    files_exists = ex.([:pm, :Δ, :Uπs, :orb, :preps])

    if !all(files_exists)
        compute_orbit_data(sett.logger, prepath(sett), sett.G, sett.S, sett.autS, radius=sett.radius)
    end

    cond1 = exists(filename(fullpath(sett), :λ))
    cond2 = exists(filename(fullpath(sett), :P))

    if !sett.warmstart && cond1 && cond2
        λ, P = PropertyT.λandP(fullpath(sett))
    else
        info(sett.logger, "Creating SDP problem...")
        SDP_problem, orb_data = create_SDP_problem(sett)
        JuMP.setsolver(SDP_problem, sett.solver)

        λ, P = λandP(SDP_problem, orb_data, sett)
    end

    info(sett.logger, "λ = $λ")
    info(sett.logger, "sum(P) = $(sum(P))")
    info(sett.logger, "maximum(P) = $(maximum(P))")
    info(sett.logger, "minimum(P) = $(minimum(P))")

    isapprox(eigvals(P), abs.(eigvals(P)), atol=sett.tol) ||
        warn("The solution matrix doesn't seem to be positive definite!")

    if λ > 0
        return check_λ(sett.name, sett.S, λ, P, sett.radius, sett.logger)
    end
    info(sett.logger, "κ($(sett.name), S) ≥ $λ < 0: Tells us nothing about property (T)")
    return false
end
