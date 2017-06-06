using JuMP
import MathProgBase: AbstractMathProgSolver

function constraints_from_pm(pm, total_length=maximum(pm))
    n = size(pm,1)
    constraints = constraints = [Array{Int,1}[] for x in 1:total_length]
    for j in 1:n
        for i in 1:n
            idx = pm[i,j]
            push!(constraints[idx], [i,j])
        end
    end
    return constraints
end

function splaplacian(RG::GroupRing, S, Id=RG.group(), n=length(basis),T::Type=Int)
    result = RG(spzeros(T, n))
    result[Id] = T(length(S))
    for s in S
        result[s] -= one(T)
    end
    return result
end

function create_SDP_problem(Δ::GroupRingElem, matrix_constraints; upper_bound=Inf)
    N = size(parent(Δ).pm, 1)
    Δ² = Δ*Δ
    @assert length(Δ.coeffs) == length(matrix_constraints)
    m = JuMP.Model();
    JuMP.@variable(m, P[1:N, 1:N])
    JuMP.@SDconstraint(m, P >= 0)
    JuMP.@constraint(m, sum(P[i] for i in eachindex(P)) == 0)

    if upper_bound < Inf
        JuMP.@variable(m, 0.0 <= λ <= upper_bound)
    else
        JuMP.@variable(m, λ >= 0)
    end

    for (pairs, δ², δ) in zip(matrix_constraints, Δ².coeffs, Δ.coeffs)
        JuMP.@constraint(m, sum(P[i,j] for (i,j) in pairs) == δ² - λ*δ)
    end

    JuMP.@objective(m, Max, λ)

    return m, λ, P
end

function solve_SDP(SDP_problem)
    info(logger, Base.repr(SDP_problem))

    # to change buffering mode of stdout to _IOLBF (line bufferin)
    # see https://github.com/JuliaLang/julia/issues/8765
    ccall((:printf, "libc"), Int, (Ptr{UInt8},), "\n");

    o = redirect_stdout(solver_logger.handlers["solver_log"].io)

    t = @timed solution_status = JuMP.solve(SDP_problem)
    info(logger, timed_msg(t))
    Base.Libc.flush_cstdio()

    redirect_stdout(o)

    if solution_status != :Optimal
        warn(logger, "The solver did not solve the problem successfully!")
    end
    info(logger, solution_status)

    return 0
end
