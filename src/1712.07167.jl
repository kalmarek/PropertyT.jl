using Printf

###############################################################################
#
#  Settings and filenames
#
###############################################################################

abstract type Settings end

struct Naive{El} <: Settings
    name::String
    G::Group
    S::Vector{El}
    radius::Int
    upper_bound::Float64
    
    solver::AbstractMathProgSolver
    warmstart::Bool
end

struct Symmetrized{El} <: Settings
    name::String
    G::Group
    S::Vector{El}
    autS::Group
    radius::Int
    upper_bound::Float64
        
    solver::AbstractMathProgSolver
    warmstart::Bool
end

function Settings(name::String,
    G::Group, S::Vector{<:GroupElem},solver::Solver;
    radius::Integer=2, upper_bound::Float64=1.0, warmstart=true) where {Solver<:AbstractMathProgSolver}
    return Naive(name, G, S, radius, upper_bound, solver, warmstart)
end

function Settings(name::String,
    G::Group, S::Vector{<:GroupElem}, autS::Group, solver::Solver;
    radius::Integer=2, upper_bound::Float64=1.0, warmstart=true) where {Solver<:AbstractMathProgSolver}
    return Symmetrized(name, G, S, autS, radius, upper_bound, solver, warmstart)
end

prefix(s::Naive) = s.name
prefix(s::Symmetrized) = "o"*s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

filename(sett::Settings, s::Symbol) = filename(sett, Val{s})

filename(sett::Settings, ::Type{Val{:fulllog}}) =
    joinpath(fullpath(sett), "full_$(string(now())).log")
filename(sett::Settings, ::Type{Val{:solverlog}}) =
    joinpath(fullpath(sett), "solver_$(string(now())).log")

filename(sett::Settings, ::Type{Val{:Δ}}) =
    joinpath(prepath(sett), "delta.jld")
filename(sett::Settings, ::Type{Val{:OrbitData}}) =
    joinpath(prepath(sett), "OrbitData.jld")

filename(sett::Settings, ::Type{Val{:warmstart}}) =
    joinpath(fullpath(sett), "warmstart.jld")
filename(sett::Settings, ::Type{Val{:solution}}) =
    joinpath(fullpath(sett), "solution.jld")

###############################################################################
#
#  λandP
#
###############################################################################

function warmstart(sett::Settings)
    if sett.warmstart && isfile(filename(sett, :warmstart))
        ws = load(filename(sett, :warmstart), "warmstart")
    else
        ws = nothing
    end
    return ws
end

function computeλandP(sett::Naive, Δ::GroupRingElem;
    solverlog=tempname()*".log")

    @info("Creating SDP problem...")
    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, upper_bound=sett.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver)
    @info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, (λ, P, ws) = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)
    @info("Solver's status: $status")
    
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

    @info("Creating SDP problem...")

    SDP_problem, varλ, varP = SOS_problem(Δ^2, Δ, orbit_data, upper_bound=sett.upper_bound)
    JuMP.setsolver(SDP_problem, sett.solver)
    @info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, (λ, Ps, ws) = PropertyT.solve(solverlog, SDP_problem, varλ, varP, ws)
    @info("Solver's status: $status")
    
    save(filename(sett, :warmstart), "warmstart", ws, "Ps", Ps, "λ", λ)
    @info("Reconstructing P...")
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
    
    @info("Computing sum of squares decomposition...")
    @time residual = eoi - compute_SOS(parent(eoi), augIdproj(Q))
    @info("ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(@sprintf("%.10f", aug(residual)))")
    L1_norm = norm(residual,1)
    @info("‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(@sprintf("%.10f", L1_norm))")

    distance = λ - 2.0^(2ceil(log2(R)))*L1_norm

    @info("Floating point distance (to positive cone) ≈")
    @info("$(@sprintf("%.10f", distance))")

    if distance ≤ 0
        return distance
    end

    @info("-"^76)
    @info("Checking in interval arithmetic...")
    λ = @interval(λ)
    @info("λ ∈ $λ")
    eoi = Δ^2 - λ*Δ
    
    @info("Projecting columns of Q to the augmentation ideal...")
    @time Q, check = augIdproj(Interval, Q)
    @info("Checking that sum of every column contains 0.0... ")
    @info((check ? "They do." : "FAILED!"))
    check || @warn("The following numbers are meaningless!")
    
    @info("Computing sum of squares decomposition...")
    @time residual = eoi - compute_SOS(parent(eoi), Q)
    @info("ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ∈ $(aug(residual))")
    L1_norm = norm(residual,1)
    @info("‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(L1_norm)")

    distance = λ - 2.0^(2ceil(log2(R)))*L1_norm
    
    @info("Interval distance (to positive cone) ∈")
    @info("$(distance)")
    @info("-"^76)

    return distance.lo
end

###############################################################################
#
#  Interpreting the numerical results
#
###############################################################################

Kazhdan(λ::Number, N::Integer) = sqrt(2*λ/N)

function interpret_results(sett::Settings, sgap::Number)

    if sgap > 0
        Kazhdan_κ = Kazhdan(sgap, length(sett.S))
        if Kazhdan_κ > 0
            @info("κ($(sett.name), S) ≥ $Kazhdan_κ: Group HAS property (T)!")
            return true
        end
    end
    @info("λ($(sett.name), S) ≥ $sgap < 0: Tells us nothing about property (T)")
    return false
end

function check_property_T(sett::Settings)
    fp = PropertyT.fullpath(sett)
    isdir(fp) || mkpath(fp)
    @info("="^76)
    @info("Running tests for $(sett.name):")
    @info("Upper bound for λ: $(sett.upper_bound), on radius $(sett.radius).")
    @info("Solver is $(sett.solver)")
    @info("Warmstart: $(sett.warmstart)")
    @info("="^76)

    if isfile(filename(sett,:Δ))
        # cached
        @info("Loading precomputed Δ...")
        Δ = loadGRElem(filename(sett,:Δ), sett.G)
    else
        # compute
        Δ = Laplacian(sett.S, sett.radius)
        saveGRElem(filename(sett, :Δ), Δ)
    end

    if !sett.warmstart && isfile(filename(sett, :solution))
        λ, P = load(filename(sett, :solution), "λ", "P")
    else
        λ, P = computeλandP(sett, Δ,
            solverlog=filename(sett, :solverlog))

        save(filename(sett, :solution), "λ", λ, "P", P)

        if λ < 0
            @warn("Solver did not produce a valid solution!")
        end
    end

    @info("λ = $λ")
    @info("sum(P) = $(sum(P))")
    @info("maximum(P) = $(maximum(P))")
    @info("minimum(P) = $(minimum(P))")

    isapprox(eigvals(P), abs.(eigvals(P))) ||
        @warn("The solution matrix doesn't seem to be positive definite!")

    @time Q = real(sqrt( (P.+ P')./2 ))
    sgap = distance_to_positive_cone(Δ, λ, Q, R=sett.radius)

    return interpret_results(sett, sgap)
end
