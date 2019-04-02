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
    
    solver::JuMP.OptimizerFactory
    warmstart::Bool
end

struct Symmetrized{El} <: Settings
    name::String
    G::Group
    S::Vector{El}
    autS::Group
    radius::Int
    upper_bound::Float64
        
    solver::JuMP.OptimizerFactory
    warmstart::Bool
end

function Settings(name::String,
    G::Group, S::Vector{<:GroupElem}, solver::JuMP.OptimizerFactory;
    radius::Integer=2, upper_bound::Float64=1.0, warmstart=true)
    return Naive(name, G, S, radius, upper_bound, solver, warmstart)
end

function Settings(name::String,
    G::Group, S::Vector{<:GroupElem}, autS::Group, solver::JuMP.OptimizerFactory;
    radius::Integer=2, upper_bound::Float64=1.0, warmstart=true)
    return Symmetrized(name, G, S, autS, radius, upper_bound, solver, warmstart)
end

prefix(s::Naive) = s.name
prefix(s::Symmetrized) = "o"*s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

filename(sett::Settings, s::Symbol; kwargs...) = filename(sett, Val{s}; kwargs...)

filename(sett::Settings, ::Type{Val{:fulllog}}) =
    joinpath(fullpath(sett), "full_$(string(now())).log")
filename(sett::Settings, ::Type{Val{:solverlog}}) =
    joinpath(fullpath(sett), "solver_$(string(now())).log")

filename(sett::Settings, ::Type{Val{:Δ}}) =
    joinpath(prepath(sett), "delta.jld")
filename(sett::Settings, ::Type{Val{:OrbitData}}) =
    joinpath(prepath(sett), "OrbitData.jld")

filename(sett::Settings, ::Type{Val{:solution}}) =
        joinpath(fullpath(sett), "solution.jld")

function filename(sett::Settings, ::Type{Val{:warmstart}}; date=false)
    if date
        return joinpath(fullpath(sett), "warmstart_$(Dates.now()).jld")
    else
        return joinpath(fullpath(sett), "warmstart.jld")
    end
end

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

    @info "Creating SDP problem..."
    SDP_problem = SOS_problem(Δ^2, Δ, upper_bound=sett.upper_bound)
    @info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, ws = PropertyT.solve(solverlog, SDP_problem, sett.solver, ws)
    @info "Optimization has finished:" status
    
    P = value.(SDP_problem[:P])
    λ = value(SDP_problem[:λ])
    
    if any(isnan.(P))
        @warn "The solution seems to contain NaNs. Not overriding warmstart.jld"
    else
        save(filename(sett, :warmstart), "warmstart", (ws.primal, ws.dual, ws.slack), "P", P, "λ", λ)
    end
    
    save(filename(sett, :warmstart, date=true),
        "warmstart", (ws.primal, ws.dual, ws.slack), "P", P, "λ", λ)

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

    @info "Creating SDP problem..."
    SDP_problem, varP = SOS_problem(Δ^2, Δ, orbit_data, upper_bound=sett.upper_bound)
    @info(Base.repr(SDP_problem))

    ws = warmstart(sett)
    @time status, ws = PropertyT.solve(solverlog, SDP_problem, sett.solver, ws)
    @info "Optimization has finished:" status
    
    λ = value(SDP_problem[:λ])
    Ps = [value.(P) for P in varP]
    
    if any(any(isnan.(P)) for P in Ps)
        @warn "The solution seems to contain NaNs. Not overriding warmstart.jld"
    else
        save(filename(sett, :warmstart), "warmstart", (ws.primal, ws.dual, ws.slack), "Ps", Ps, "λ", λ)
    end
    
    save(filename(sett, :warmstart, date=true), 
        "warmstart", (ws.primal, ws.dual, ws.slack), "Ps", Ps, "λ", λ)
    
    @info "Reconstructing P..."
    @time P = reconstruct(Ps, orbit_data)

    return λ, P
end

###############################################################################
#
#  Checking solution
#
###############################################################################

function distance_to_positive_cone(Δ::GroupRingElem, λ, Q; R::Int=2)
    separator = "-"^76
    @info "$separator\nChecking in floating-point arithmetic..." λ
    eoi = Δ^2-λ*Δ
    
    @info("Computing sum of squares decomposition...")
    @time residual = eoi - compute_SOS(parent(eoi), augIdproj(Q))

    L1_norm = norm(residual,1)
    distance = λ - 2.0^(2ceil(log2(R)))*L1_norm
    
    info_strs = ["Numerical metrics of the obtained SOS:",
        "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ≈ $(aug(residual))",
        "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ≈ $(L1_norm)",
        "Floating point distance (to positive cone) ≈"]
    @info join(info_strs, "\n") distance

    if distance ≤ 0
        return distance
    end

    λ = @interval(λ)
    info_strs = [separator,
        "Checking in interval arithmetic...",
        "λ ∈ $λ"]
    @info(join(info_strs, "\n"))
    eoi = Δ^2 - λ*Δ
    
    @info("Projecting columns of Q to the augmentation ideal...")
    @time Q, check = augIdproj(Interval, Q)
    result = (check ? "Correct." : "FAILED!")
    @info "Checking that sum of every column contains 0.0..." result
    check || @warn("The following numbers are meaningless!")
    
    @info("Computing sum of squares decomposition...")
    @time residual = eoi - compute_SOS(parent(eoi), Q)
    
    L1_norm = norm(residual,1)
    distance = λ - 2.0^(2ceil(log2(R)))*L1_norm
    
    info_strs = ["Numerical metrics of the obtained SOS:",
        "ɛ(Δ² - λΔ - ∑ξᵢ*ξᵢ) ∈ $(aug(residual))",
        "‖Δ² - λΔ - ∑ξᵢ*ξᵢ‖₁ ∈ $(L1_norm)",
        "Interval distance (to positive cone) ∈"]
    @info join(info_strs, "\n") distance

    return distance.lo
end

###############################################################################
#
#  Interpreting the numerical results
#
###############################################################################

Kazhdan(λ::Number, N::Integer) = sqrt(2*λ/N)

function check_property_T(sett::Settings)
    print_summary(sett)
    certified_sgap = spectral_gap(sett)
    return interpret_results(sett, certified_sgap)
end

function print_summary(sett::Settings)
    separator = "="^76
    info_strs = [separator,
    "Running tests for $(sett.name):",
    "Upper bound for λ: $(sett.upper_bound), on radius $(sett.radius).",
    "Warmstart: $(sett.warmstart)",
    "Results will be stored in ./$(PropertyT.prepath(sett))",
    "Solver: $(typeof(sett.solver()))",
    "Solvers options: "]
    append!(info_strs, [rpad("   $k", 30)* "→ \t$v" for (k,v) in sett.solver().options])
    push!(info_strs, separator)
    @info join(info_strs, "\n")
end

function interpret_results(sett::Settings, sgap::Number)
    if sgap > 0
        Kazhdan_κ = Kazhdan(sgap, length(sett.S))
        if Kazhdan_κ > 0
            @info "κ($(sett.name), S) ≥ $Kazhdan_κ: Group HAS property (T)!"
            return true
        end
    end
    info_strs = ["The certified lower bound on the spectral gap is negative:",
        "λ($(sett.name), S) ≥ 0.0 > $sgap",
        "This tells us nothing about property (T)"]
    @info join(info_strs, "\n")
    return false
end

function spectral_gap(sett::Settings)
    fp = PropertyT.fullpath(sett)
    isdir(fp) || mkpath(fp)
    
    if isfile(filename(sett,:Δ))
        # cached
        @info "Loading precomputed Δ..."
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
            @warn "Solver did not produce a valid solution!"
        end
    end
    
    info_strs = ["Numerical metrics of matrix solution:",
        "sum(P) = $(sum(P))",
        "maximum(P) = $(maximum(P))",
        "minimum(P) = $(minimum(P))"]
    @info join(info_strs, "\n")

    isapprox(eigvals(P), abs.(eigvals(P))) ||
        @warn "The solution matrix doesn't seem to be positive definite!"

    @time Q = real(sqrt(Symmetric( (P.+ P')./2 )))
    certified_sgap = distance_to_positive_cone(Δ, λ, Q, R=sett.radius)
    
    return certified_sgap
end
