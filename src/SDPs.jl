###############################################################################
#
#  Constraints
#
###############################################################################

function constraints(pm::Matrix{I}, total_length=maximum(pm)) where {I<:Integer}
    cnstrs = [Vector{I}() for _ in 1:total_length]
    for i in eachindex(pm)
        push!(cnstrs[pm[i]], i)
    end
    return cnstrs
end

function orbit_constraint!(result::SparseMatrixCSC, cnstrs, orbit; val=1.0/length(orbit))
    result .= zero(eltype(result))
    dropzeros!(result)
    for constraint in cnstrs[orbit]
        for idx in constraint
            result[idx] = val
        end
    end
    return result
end

###############################################################################
#
#  Naive SDP
#
###############################################################################

function SOS_problem(X::GroupRingElem, orderunit::GroupRingElem; upper_bound=Inf)
    N = size(parent(X).pm, 1)
    m = JuMP.Model();

    JuMP.@variable(m, P[1:N, 1:N])
    JuMP.@SDconstraint(m, P >= 0)
    JuMP.@constraint(m, sum(P[i] for i in eachindex(P)) == 0)

    JuMP.@variable(m, λ)
    if upper_bound < Inf
        JuMP.@constraint(m, λ <= upper_bound)
    end

    cnstrs = constraints(parent(X).pm)

    for (constraint, x, u) in zip(cnstrs, X.coeffs, orderunit.coeffs)
        JuMP.@constraint(m, sum(P[constraint]) == x - λ*u)
    end

    JuMP.@objective(m, Max, λ)
    return m, λ, P
end

###############################################################################
#
#  Symmetrized SDP
#
###############################################################################

function SOS_problem(X::GroupRingElem, orderunit::GroupRingElem, data::OrbitData; upper_bound=Inf)
    Ns = size.(data.Uπs, 2)
    m = JuMP.Model();

    P = Vector{Matrix{JuMP.Variable}}(length(Ns))

    for (k,s) in enumerate(Ns)
        P[k] = JuMP.@variable(m, [i=1:s, j=1:s])
        JuMP.@SDconstraint(m, P[k] >= 0.0)
    end

    λ = JuMP.@variable(m, λ)
    if upper_bound < Inf
        JuMP.@constraint(m, λ <= upper_bound)
    end

    info("Adding $(length(data.orbits)) constraints... ")

    @time addconstraints!(m,P,λ,X,orderunit, data)

    JuMP.@objective(m, Max, λ)
    return m, λ, P
end

function constraintLHS!(M, cnstr, Us, Ust, dims, eps=1000*eps(eltype(first(M))))
    for π in 1:endof(Us)
        M[π] = PropertyT.sparsify!(dims[π].*Ust[π]*cnstr*Us[π], eps)
    end
end

function addconstraints!(m::JuMP.Model,
    P::Vector{Matrix{JuMP.Variable}}, λ::JuMP.Variable,
    X::GroupRingElem, orderunit::GroupRingElem, data::OrbitData)

    orderunit_orb = orbit_spvector(orderunit.coeffs, data.orbits)
    X_orb = orbit_spvector(X.coeffs, data.orbits)
    UπsT = [U' for U in data.Uπs]

    cnstrs = constraints(parent(X).pm)
    orb_cnstr = spzeros(Float64, size(parent(X).pm)...)

    M = [Array{Float64}(n,n) for n in size.(UπsT,1)]

    for (t, orbit) in enumerate(data.orbits)
        orbit_constraint!(orb_cnstr, cnstrs, orbit)
        constraintLHS!(M, orb_cnstr, data.Uπs, UπsT, data.dims)

        lhs = @expression(m, sum(vecdot(M[π], P[π]) for π in 1:endof(data.Uπs)))
        x, u = X_orb[t], orderunit_orb[t]
        JuMP.@constraint(m, lhs == x - λ*u)
    end
end

function reconstruct(Ps::Vector{Matrix{F}}, data::OrbitData) where F
    return reconstruct(Ps, data.preps, data.Uπs, data.dims)
end

function reconstruct(Ps::Vector{M},
    preps::Dict{GEl, P}, Uπs::Vector{U}, dims::Vector{Int}) where
        {M<:AbstractMatrix, GEl<:GroupElem, P<:perm, U<:AbstractMatrix}

    l = length(Uπs)
    transfP = [dims[π].*Uπs[π]*Ps[π]*Uπs[π]' for π in 1:l]
    tmp = [zeros(Float64, size(first(transfP))) for _ in 1:l]
    perms = collect(keys(preps))

    Threads.@threads for π in 1:l
        for p in perms
            BLAS.axpy!(1.0, view(transfP[π], preps[p].d, preps[p].d), tmp[π])
        end
    end

    recP = 1/length(perms) .* sum(tmp)
    # for i in eachindex(recP)
    #     if abs(recP[i]) .< eps(eltype(recP))*100
    #         recP[i] = zero(eltype(recP))
    #     end
    # end
    return recP
end

###############################################################################
#
#  Low-level solve
#
###############################################################################
using MathProgBase

function solve(m::JuMP.Model, varλ::JuMP.Variable, varP; warmstart=nothing)

    traits = JuMP.ProblemTraits(m, relaxation=false)

    JuMP.build(m, traits=traits)
    if warmstart != nothing
        p_sol, d_sol, s = warmstart
        MathProgBase.SolverInterface.setwarmstart!(m.internalModel, p_sol;
            dual_sol=d_sol, slack=s);
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

function solve_logged(model::JuMP.Model, varλ::JuMP.Variable, varP, warmstart=nothing; solverlog::String=tempname()*".log")

    function f()
        Base.Libc.flush_cstdio()
        λ, P, ws = PropertyT.solve(model, varλ, varP, warmstart=warmstart)
        Base.Libc.flush_cstdio()
        return λ, P, ws
    end

    isdir(dirname(solverlog)) || mkpath(dirname(solverlog))

    log = open(solverlog, "a+")
    λ, P, warmstart = redirect_stdout(f, log)
    close(log)

    return λ, P, warmstart
end

###############################################################################
#
#  Copied from JuMP/src/solvers.jl:178
#
###############################################################################

function fillfrominternal!(m::JuMP.Model, traits)

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
