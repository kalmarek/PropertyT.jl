using JuMP
import MathProgBase: AbstractMathProgSolver

function constraints(pm, total_length=maximum(pm))
    n = size(pm,1)
    constraints = [Vector{Tuple{Int,Int}}() for _ in 1:total_length]
    for j in 1:n
        for i in 1:n
            idx = pm[i,j]
            push!(constraints[idx], (i,j))
        end
    end
    return constraints
end

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

function SOS_problem(X::GroupRingElem, orderunit::GroupRingElem; upper_bound=Inf)
    N = size(parent(X).pm, 1)
    matrix_constraints = PropertyT.constraints(parent(X).pm)
    m = JuMP.Model();

    JuMP.@variable(m, P[1:N, 1:N])
    JuMP.@SDconstraint(m, P >= 0)
    JuMP.@constraint(m, sum(P[i] for i in eachindex(P)) == 0)

    JuMP.@variable(m, λ)
    if upper_bound < Inf
        JuMP.@constraint(m, λ <= upper_bound)
    end

    for (cnstr, x, u) in zip(matrix_constraints, X.coeffs, orderunit.coeffs)
        JuMP.@constraint(m, sum(P[cnstr]) == x - λ*u)
    end

    JuMP.@objective(m, Max, λ)
    return m, λ, P
end

function solve_SDP(model::JuMP.Model, varλ, varP, ws=nothing; solverlog=tempname()*".log")

    function f()
        Base.Libc.flush_cstdio()
        λ, P, ws = PropertyT.solve(model, varλ, varP, warmstart=ws)
        Base.Libc.flush_cstdio()
        return λ, P, ws
    end

    log = open(solverlog, "a+")
    λ, P, ws = redirect_stdout(f, log)
    close(log)

    return λ, P, ws
end

function solve(m::JuMP.Model, varλ, varP; warmstart=nothing)

    traits = JuMP.ProblemTraits(m, relaxation=false)

    JuMP.build(m, traits=traits)
    if warmstart != nothing
        p_sol, d_sol, s = warmstart
        MathProgBase.SolverInterface.setwarmstart!(m.internalModel, p_sol; dual_sol = d_sol, slack=s);
    end

    MathProgBase.optimize!(m.internalModel)

    λ = MathProgBase.getobjval(m.internalModel)

    warmstart = (m.internalModel.primal_sol, m.internalModel.dual_sol,
          m.internalModel.slack)

    fillfrominternal!(m, traits)

    P = JuMP.getvalue(varP)
    λ = JuMP.getvalue(varλ)

    return λ, P, warmstart
end

function fillfrominternal!(m::JuMP.Model, traits)
    # Copied from JuMP/src/solvers.jl:178

    stat::Symbol = MathProgBase.status(m.internalModel)

    numRows, numCols = length(m.linconstr), m.numCols
    m.objBound = NaN
    m.objVal = NaN
    m.colVal = fill(NaN, numCols)
    m.linconstrDuals = Array{Float64}(0)

    discrete = (traits.int || traits.sos)

    if stat == :Optimal
        # If we think dual information might be available, try to get it
        # If not, return an array of the correct length
        if discrete
            m.redCosts = fill(NaN, numCols)
            m.linconstrDuals = fill(NaN, numRows)
        else
            if !traits.conic
                m.redCosts = try
                    MathProgBase.getreducedcosts(m.internalModel)[1:numCols]
                catch
                    fill(NaN, numCols)
                end

                m.linconstrDuals = try
                    MathProgBase.getconstrduals(m.internalModel)[1:numRows]
                catch
                    fill(NaN, numRows)
                end
            elseif !traits.qp && !traits.qc
                JuMP.fillConicDuals(m)
            end
        end
    else
        # Problem was not solved to optimality, attempt to extract useful
        # information anyway

        if traits.lin
            if stat == :Infeasible
                m.linconstrDuals = try
                    infray = MathProgBase.getinfeasibilityray(m.internalModel)
                    @assert length(infray) == numRows
                    infray
                catch
                    suppress_warnings || warn("Infeasibility ray (Farkas proof) not available")
                    fill(NaN, numRows)
                end
            elseif stat == :Unbounded
                m.colVal = try
                    unbdray = MathProgBase.getunboundedray(m.internalModel)
                    @assert length(unbdray) == numCols
                    unbdray
                catch
                    suppress_warnings || warn("Unbounded ray not available")
                    fill(NaN, numCols)
                end
            end
        end
        # conic duals (currently, SOC and SDP only)
        if !discrete && traits.conic && !traits.qp && !traits.qc
            if stat == :Infeasible
                JuMP.fillConicDuals(m)
            end
        end
    end

    # If the problem was solved, or if it terminated prematurely, try
    # to extract a solution anyway. This commonly occurs when a time
    # limit or tolerance is set (:UserLimit)
    if !(stat == :Infeasible || stat == :Unbounded)
        try
            # Do a separate try since getobjval could work while getobjbound does not and vice versa
            objBound = MathProgBase.getobjbound(m.internalModel) + m.obj.aff.constant
            m.objBound = objBound
        end
        try
            objVal = MathProgBase.getobjval(m.internalModel) + m.obj.aff.constant
            colVal = MathProgBase.getsolution(m.internalModel)[1:numCols]
            # Rescale off-diagonal terms of SDP variables
            if traits.sdp
                offdiagvars = JuMP.offdiagsdpvars(m)
                colVal[offdiagvars] /= sqrt(2)
            end
            # Don't corrupt the answers if one of the above two calls fails
            m.objVal = objVal
            m.colVal = colVal
        end
    end

    return stat
end
