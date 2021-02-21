###############################################################################
#  Settings and filenames

abstract type Settings end

struct Naive{El} <: Settings
    name::String
    G::Union{Group, NCRing}
    S::Vector{El}
    halfradius::Int
    upper_bound::Float64

    solver::JuMP.OptimizerFactory
    force_compute::Bool
end

struct Symmetrized{El} <: Settings
    name::String
    G::Union{Group, NCRing}
    S::Vector{El}
    autS::Group
    halfradius::Int
    upper_bound::Float64

    solver::JuMP.OptimizerFactory
    force_compute::Bool
end

function Settings(name::String,
    G::Union{Group, NCRing}, S::AbstractVector{El}, solver::JuMP.OptimizerFactory;
    halfradius=2, upper_bound=1.0, force_compute=false) where El <: Union{GroupElem, NCRingElem}
    return Naive(name, G, S, halfradius, upper_bound, solver, force_compute)
end

function Settings(name::String,
    G::Union{Group, NCRing}, S::AbstractVector{El}, autS::Group, solver::JuMP.OptimizerFactory;
    halfradius=2, upper_bound=1.0, force_compute=false) where El <: Union{GroupElem, NCRingElem}
    return Symmetrized(name, G, S, autS, halfradius, upper_bound, solver, force_compute)
end

suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = s.name
fullpath(s::Settings) = joinpath(prepath(s), suffix(s))

filename(sett::Settings, s::Symbol; kwargs...) = filename(sett, Val{s}; kwargs...)

filename(sett::Settings, ::Type{Val{:fulllog}}; kwargs...) =
    filename(fullpath(sett), "full", "log", suffix=Dates.now(); kwargs...)
filename(sett::Settings, ::Type{Val{:solverlog}}; kwargs...) =
    filename(fullpath(sett), "solver", "log", suffix=Dates.now(); kwargs...)

filename(sett::Settings, ::Type{Val{:Δ}}; kwargs...) =
    filename(prepath(sett), "delta", "jld"; kwargs...)
filename(sett::Settings, ::Type{Val{:BlockDecomposition}}; kwargs...) =
    filename(prepath(sett), "BlockDecomposition", "jld"; kwargs...)

filename(sett::Settings, ::Type{Val{:solution}}; kwargs...) =
    filename(fullpath(sett), "solution", "jld"; kwargs...)

function filename(sett::Settings, ::Type{Val{:warmstart}}; kwargs...)
    filename(fullpath(sett), "warmstart", "jld"; kwargs...)
end

function filename(path::String, name, extension; prefix=nothing, suffix=nothing)
    pre = isnothing(prefix) ? "" : "$(prefix)_"
    suf = isnothing(suffix) ? "" : "_$(suffix)"
    return joinpath(path, "$pre$name$suf.$extension")
end

###############################################################################
#  Approximation by SOS (logged & warmstarted)

function warmstart(sett::Settings)
    warmstart_fname = filename(sett, :warmstart)
    try
        ws = load(warmstart_fname, "warmstart")
        @info "Loaded $warmstart_fname."
        return ws
    catch ex
        @warn "$(ex.msg). Could not provide a warmstart to the solver."
        return nothing
    end
end

function approximate_by_SOS(sett::Naive,
    elt::GroupRingElem, orderunit::GroupRingElem;
    solverlog=tempname()*".log")

    isdir(fullpath(sett)) || mkpath(fullpath(sett))

    @info "Creating SDP problem..."
    SDP_problem = SOS_problem_primal(elt, orderunit, upper_bound=sett.upper_bound)
    @info Base.repr(SDP_problem)

    @info "Logging solver's progress into $solverlog"

    ws = warmstart(sett)
    @time status, ws = PropertyT.solve(solverlog, SDP_problem, sett.solver, ws)
    @info "Optimization finished:" status

    P = value.(SDP_problem[:P])
    λ = value(SDP_problem[:λ])

    if any(isnan, P)
        @warn "The solution seems to contain NaNs. Not overriding warmstart.jld"
    else
        save(filename(sett, :warmstart),
            "warmstart", (ws.primal, ws.dual, ws.slack),
            "P", P,
            "λ", λ)
    end

    save(filename(sett, :warmstart, suffix=Dates.now()),
        "warmstart", (ws.primal, ws.dual, ws.slack),
        "P", P,
        "λ", λ)

    return λ, P
end

function approximate_by_SOS(sett::Symmetrized,
    elt::GroupRingElem, orderunit::GroupRingElem;
    solverlog=tempname()*".log")

    isdir(fullpath(sett)) || mkpath(fullpath(sett))

    orbit_data = try
        orbit_data = load(filename(sett, :BlockDecomposition), "BlockDecomposition")
        @info "Loaded orbit data."
        orbit_data
    catch ex
        @warn ex.msg
        GroupRings.hasbasis(parent(orderunit)) ||
            throw("You need to define basis of Group Ring to compute orbit decomposition!")
        @info "Computing orbit and Wedderburn decomposition..."
        orbit_data = BlockDecomposition(parent(orderunit), sett.autS)
        save(filename(sett, :BlockDecomposition), "BlockDecomposition", orbit_data)
        orbit_data
    end

    orbit_data = decimate(orbit_data)

    @info "Creating SDP problem..."
    SDP_problem, varP = SOS_problem_primal(elt, orderunit, orbit_data, upper_bound=sett.upper_bound)
    @info Base.repr(SDP_problem)

    @info "Logging solver's progress into $solverlog"

    ws = warmstart(sett)
    @time status, ws = PropertyT.solve(solverlog, SDP_problem, sett.solver, ws)
    @info "Optimization finished:" status

    λ = value(SDP_problem[:λ])
    Ps = [value.(P) for P in varP]

    if any(any(isnan, P) for P in Ps)
        @warn "The solution seems to contain NaNs. Not overriding warmstart.jld"
    else
        save(filename(sett, :warmstart),
            "warmstart", (ws.primal, ws.dual, ws.slack),
            "Ps", Ps,
            "λ",  λ)
    end

    save(filename(sett, :warmstart, suffix=Dates.now()),
        "warmstart", (ws.primal, ws.dual, ws.slack),
        "Ps", Ps,
        "λ",  λ)

    @info "Reconstructing P..."
    @time P = reconstruct(Ps, orbit_data)

    return λ, P
end

###############################################################################
#  Checking solution

function certify_SOS_decomposition(elt::GroupRingElem, orderunit::GroupRingElem,
    λ::Number, Q::AbstractMatrix; R::Int=2)
    separator = "-"^76
    @info "$separator\nChecking in floating-point arithmetic..." λ
    eoi = elt - λ*orderunit

    @info("Computing sum of squares decomposition...")
    @time residual = eoi - compute_SOS(parent(eoi), augIdproj(Q))

    L1_norm = norm(residual,1)
    floatingpoint_λ = λ - 2.0^(2ceil(log2(R)))*L1_norm

    info_strs = ["Numerical metrics of the obtained SOS:",
        "ɛ(elt - λu - ∑ξᵢ*ξᵢ) ≈ $(aug(residual))",
        "‖elt - λu - ∑ξᵢ*ξᵢ‖₁ ≈ $(L1_norm)",
        "Floating point (NOT certified) λ ≈"]
    @info join(info_strs, "\n") floatingpoint_λ

    if floatingpoint_λ ≤ 0
        return floatingpoint_λ
    end

    λ = @interval(λ)
    info_strs = [separator,
        "Checking in interval arithmetic...",
        "λ ∈ $λ"]
    @info(join(info_strs, "\n"))
    eoi = elt - λ*orderunit

    @info("Projecting columns of Q to the augmentation ideal...")
    @time Q, check = augIdproj(Interval, Q)
    @info "Checking that sum of every column contains 0.0..." check_augmented=check
    check || @error("The following numbers are meaningless!")

    @info("Computing sum of squares decomposition...")
    @time residual = eoi - compute_SOS(parent(eoi), Q)

    L1_norm = norm(residual,1)
    certified_λ = λ - 2.0^(2ceil(log2(R)))*L1_norm

    info_strs = ["Numerical metrics of the obtained SOS:",
        "ɛ(elt - λu - ∑ξᵢ*ξᵢ) ∈ $(aug(residual))",
        "‖elt - λu - ∑ξᵢ*ξᵢ‖₁ ∈ $(L1_norm)",
        "Interval aritmetic (certified) λ ∈"]
    @info join(info_strs, "\n") certified_λ

    return certified_λ
end

function spectral_gap(Δ::GroupRingElem, λ::Number, Q::AbstractMatrix; R::Int=2)
    @info "elt = Δ², u = Δ"
    return certify_SOS_decomposition(Δ^2, Δ, λ, Q, R=R)
end

###############################################################################
#  Interpreting the numerical results

Kazhdan_constant(λ::Number, N::Integer) = sqrt(2*λ/N)
Kazhdan_constant(λ::Interval, N::Integer) = IntervalArithmetic.inf(sqrt(2*λ/N))

function check_property_T(sett::Settings)
    @info sett
    certified_sgap = spectral_gap(sett)
    return interpret_results(sett, certified_sgap)
end

function Base.show(io::IO, sett::Settings)
    info_strs = ["PropertyT Settings:",
    "Group: $(sett.name)",
    "Upper bound for λ: $(sett.upper_bound), on halfradius $(sett.halfradius).",
    "Force computations: $(sett.force_compute);",
    "Results will be stored in ./$(PropertyT.prepath(sett));",
    "Solver: $(typeof(sett.solver()))",
    "Solvers options: "]
    append!(info_strs, [rpad("   $k", 30)* "→ \t$v" for (k,v) in sett.solver().options])
    push!(info_strs, "="^76)
    print(io, join(info_strs, "\n"))
end

function interpret_results(name::String, sgap::Number, N::Integer)
    if sgap > 0
        κ = Kazhdan_constant(sgap, N)
        @info "κ($name, S) ≥ $κ: Group HAS property (T)!"
        return true
    end
    info_strs = [
        "The certified lower bound on the spectral gap is negative:",
        "λ($name, S) ≥ 0.0 > $sgap",
        "This tells us nothing about property (T)",
    ]
    @info join(info_strs, "\n")
    return false
end

interpret_results(sett::Settings, sgap::Number) =
    interpret_results(sett.name, sgap, length(sett.S))

function spectral_gap(sett::Settings)
    fp = PropertyT.fullpath(sett)
    isdir(fp) || mkpath(fp)

    Δ = try
        Δ = loadGRElem(filename(sett,:Δ), sett.G)
        @info "Loaded precomputed Δ."
        Δ
    catch ex
        @warn ex.msg
        @info "Computing Δ..."
        Δ = Laplacian(sett.S, sett.halfradius)
        saveGRElem(filename(sett, :Δ), Δ)
        Δ
    end

    function compute(sett, Δ)
        @info "Computing λ and P..."
        λ, P = approximate_by_SOS(sett, Δ^2, Δ;
            solverlog=filename(sett, :solverlog))

        save(filename(sett, :solution), "λ", λ, "P", P)

        λ < 0 && @warn "Solver did not produce a valid solution!"
        return λ, P
    end

    if sett.force_compute
        λ, P = compute(sett, Δ)
    else
        λ, P =try
            λ, P = load(filename(sett, :solution), "λ", "P")
            @info "Loaded existing λ and P."
            λ, P
        catch ex
            @warn ex.msg
            compute(sett, Δ)
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
    certified_sgap = spectral_gap(Δ, λ, Q, R=sett.halfradius)

    return certified_sgap
end
