###############################################################################
#
#  Settings and filenames
#
###############################################################################

struct Symmetrize end
struct Naive end

abstract type PropertyTSettings end

struct SolverSettings
    sdpsolver::AbstractMathProgSolver
    upper_bound::Float64
    warmstart::Bool
    
    SolverSettings(sol, ub, ws=true) = new(sol, upper_bound, ws)
end

struct Naive <: PropertyTSettings
    name::String
    G::Group
    S::Vector{GroupElem}
    radius::Int
    
    solver::SolverSettings
end

struct Symmetrized <: PropertyTSettings
    name::String
    G::Group
    S::Vector{GroupElem}
    autS::Group
    radius::Int
    
    solver::SolverSettings
end

function Settings(name::String,
    G::Group, S::Vector{GEl}, r::Integer,
    sol::Solver, ub, ws=true) where {GEl<:GroupElem, Solver<:AbstractMathProgSolver}
    sol_sett = SolverSettings(sol, ub, ws)
    return Naive(name, G, S, r, sol_sett)
end

function Settings(name::String,
    G::Group, S::Vector{GEl}, autS::Group, r::Integer,
    sol::Solver, ub, ws=true) where {GEl<:GroupElem, Solver<:AbstractMathProgSolver}
    sol_sett = SolverSettings(sol, ub, ws)
    return Symmetrized(name, G, S, autS, r, sol_sett)
end

prefix(s::Naive) = s.name
prefix(s::Symmetrized) = "o"*s.name
suffix(s::PropertyTSettings) = "$(s.upper_bound)"
prepath(s::PropertyTSettings) = prefix(s)
fullpath(s::PropertyTSettings) = joinpath(prefix(s), suffix(s))

filename(sett::PropertyTSettings, s::Symbol) = filename(sett, Val{s})

filename(sett::PropertyTSettings, ::Type{Val{:fulllog}}) =
    joinpath(fullpath(sett), "full_$(string(now())).log")
filename(sett::PropertyTSettings, ::Type{Val{:solverlog}}) =
    joinpath(fullpath(sett), "solver_$(string(now())).log")

filename(sett::PropertyTSettings, ::Type{Val{:Δ}}) =
    joinpath(prepath(sett), "delta.jld")
filename(sett::PropertyTSettings, ::Type{Val{:OrbitData}}) =
    joinpath(prepath(sett), "OrbitData.jld")

filename(sett::PropertyTSettings, ::Type{Val{:warmstart}}) =
    joinpath(fullpath(sett), "warmstart.jld")
filename(sett::PropertyTSettings, ::Type{Val{:solution}}) =
    joinpath(fullpath(sett), "solution.jld")

###############################################################################
#
#  λandP
#
###############################################################################

function warmstart(sett::PropertyTSettings)
    if sett.solver.warmstart && isfile(filename(sett, :warmstart))
        ws = load(filename(sett, :warmstart), "warmstart")
    else
        ws = nothing
    end
    return ws
end

function computeλandP(sett::Naive, Δ::GroupRingElem;
    solverlog=tempname()*".log")

    info("Creating SDP problem...")
    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, upper_bound=sett.solver.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver.sdpsolver)
    info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, (λ, P, ws) = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)
    @show status
    save(filename(sett, :warmstart), "warmstart", ws, "P", P, "λ", λ)

    return λ, P
end

function computeλandP(sett::Symmetrized, Δ::GroupRingElem;
    solverlog=tempname()*".log")

    if !isfile(filename(sett, :OrbitData))
        isdefined(parent(Δ), :basis) || throw("You need to define basis of Group Ring to compute orbit decomposition!")
        orbit_data = OrbitData(parent(Δ), sett.autS)
        save(filename(sett, :OrbitData), "OrbitData", orbit_data)
    end
    orbit_data = load(filename(sett, :OrbitData), "OrbitData")
    orbit_data = decimate(orbit_data)

    info("Creating SDP problem...")

    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, orbit_data, upper_bound=sett.solver.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver.sdpsolver)
    info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, (λ, Ps, ws) = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)
    @show status
    save(filename(sett, :warmstart), "warmstart", ws, "Ps", Ps, "λ", λ)

    info("Reconstructing P...")
    @time P = reconstruct(Ps, orbit_data)

    return λ, P
end

###############################################################################
#
#  Checking solution
#
###############################################################################

function distance_to_positive_cone(Δ::GroupRingElem, λ, Q; R::Int=2)
    @info("------------------------------------------------------------")
    @info("Checking in floating-point arithmetic...")
    @info("λ = $λ")
    eoi = Δ^2-λ*Δ
    
    @time residual = eoi - compute_SOS(parent(eoi), augIdproj(Q))
    @info("ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", aug(residual)))")
    L1_norm = norm(residual,1)
    @info("‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", L1_norm))")

    distance = λ - 2.0^(2ceil(log2(R)))*L1_norm

    @info("Floating point distance (to positive cone) ≈")
    @info("$(@sprintf("%.10f", distance))")
    @info("")

    if distance ≤ 0
        return distance
    end

    @info("------------------------------------------------------------")
    @info("Checking in interval arithmetic...")
    @info("λ ∈ $λ")

    λ = @interval(λ)
    eoi = Δ^2 - λ*Δ
    
    @info("Projecting columns of Q to the augmentation ideal...")
    @time Q, check = augIdproj(Interval, Q)
    @info("Checking that sum of every column contains 0.0... ")
    @info((check? "They do." : "FAILED!"))
    check || @warn("The following numbers are meaningless!")

    @time residual = eoi - compute_SOS(parent(eoi), Q)
    @info("ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ∈ $(aug(residual))")
    L1_norm = norm(residual,1)
    @info("‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(L1_norm)")

    distance = λ - 2.0^(2ceil(log2(R)))*L1_norm
    
    @info("Interval distance (to positive cone) ∈")
    @info("$(distance)")
    @info("")

    return distance.lo
end

###############################################################################
#
#  Interpreting the numerical results
#
###############################################################################

Kazhdan(λ::Number, N::Integer) = sqrt(2*λ/N)

function interpret_results(sett::PropertyTSettings, sgap::Number)

    if sgap > 0
        Kazhdan_κ = Kazhdan(sgap, length(sett.S))
        if Kazhdan_κ > 0
            info("κ($(sett.name), S) ≥ $Kazhdan_κ: Group HAS property (T)!")
            return true
        end
    end
    info("λ($(sett.name), S) ≥ $sgap < 0: Tells us nothing about property (T)")
    return false
end

function check_property_T(sett::PropertyTSettings)
    fp = PropertyT.fullpath(sett)
    isdir(fp) || mkpath(fp)

    if isfile(filename(sett,:Δ))
        # cached
        Δ = loadLaplacian(filename(sett,:Δ), sett.G)
    else
        # compute
        Δ = Laplacian(sett.S, sett.radius)
        saveLaplacian(filename(sett, :Δ), Δ)
    end

    if !sett.warmstart && isfile(filename(sett, :solution))
        λ, P = load(filename(sett, :solution), "λ", "P")
    else
        λ, P = computeλandP(sett, Δ,
            solverlog=filename(sett, :solverlog))

        save(filename(sett, :solution), "λ", λ, "P", P)

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

    @time Q = real(sqrtm((P+P')/2))
    sgap = distance_to_positive_cone(Δ, λ, Q, wlen=2*sett.radius)

    return interpret_results(sett, sgap)
end
