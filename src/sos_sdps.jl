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

function orbit_spvector(vect::AbstractVector, orbits)
    orb_vector = spzeros(length(orbits))

    for (i,o) in enumerate(orbits)
        k = vect[collect(o)]
        val = k[1]
        @assert all(k .== val)
        orb_vector[i] = val
    end

    return orb_vector
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

    P = Vector{Matrix{JuMP.Variable}}(undef, length(Ns))

    for (k,s) in enumerate(Ns)
        P[k] = JuMP.@variable(m, [i=1:s, j=1:s])
        JuMP.@SDconstraint(m, P[k] >= 0.0)
    end

    λ = JuMP.@variable(m, λ)
    if upper_bound < Inf
        JuMP.@constraint(m, λ <= upper_bound)
    end

    @info("Adding $(length(data.orbits)) constraints... ")

    @time addconstraints!(m,P,λ,X,orderunit, data)

    JuMP.@objective(m, Max, λ)
    return m, λ, P
end

function constraintLHS!(M, cnstr, Us, Ust, dims, eps=1000*eps(eltype(first(M))))
    for π in 1:lastindex(Us)
        M[π] = dims[π].*PropertyT.clamp_small!(Ust[π]*cnstr*Us[π], eps)
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

    M = [Array{Float64}(undef, n,n) for n in size.(UπsT,1)]

    for (t, orbit) in enumerate(data.orbits)
        orbit_constraint!(orb_cnstr, cnstrs, orbit)
        constraintLHS!(M, orb_cnstr, data.Uπs, UπsT, data.dims)

        lhs = @expression(m, sum(dot(M[π], P[π]) for π in eachindex(data.Uπs)))
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

    lU = length(Uπs)
    transfP = [dims[π].*Uπs[π]*Ps[π]*Uπs[π]' for π in 1:lU]
    tmp = [zeros(Float64, size(first(transfP))) for _ in 1:lU]
    
    Threads.@threads for π in 1:lU
        tmp[π] = perm_avg(tmp[π], transfP[π], values(preps))
    end

    recP = sum(tmp)./length(preps)

    return recP
end

function perm_avg(result, P, perms)
    lp = length(first(perms).d)
    for p in perms
        # result .+= view(P, p.d, p.d)
        @inbounds for j in 1:lp
            k = p[j]
            for i in 1:lp
                result[i,j] += P[p[i], k]
            end
        end
    end
    return result
end

###############################################################################
#
#  Low-level solve
#
###############################################################################
using MathProgBase

function solve(m::JuMP.Model, varλ::JuMP.Variable, varP, warmstart=nothing)

    traits = JuMP.ProblemTraits(m, relaxation=false)

    JuMP.build(m, traits=traits)
    if warmstart != nothing
        p_sol, d_sol, s = warmstart
        MathProgBase.SolverInterface.setwarmstart!(m.internalModel, p_sol;
            dual_sol=d_sol, slack=s);
    end

    MathProgBase.optimize!(m.internalModel)
    status = MathProgBase.status(m.internalModel)

    λ = MathProgBase.getobjval(m.internalModel)

    warmstart = (m.internalModel.primal_sol, m.internalModel.dual_sol,
          m.internalModel.slack)

    fillfrominternal!(m, traits)

    P = JuMP.getvalue(varP)
    λ = JuMP.getvalue(varλ)

    return status, (λ, P, warmstart)
end

function solve(solverlog::String, model::JuMP.Model, varλ::JuMP.Variable, varP, warmstart=nothing)

    isdir(dirname(solverlog)) || mkpath(dirname(solverlog))

    Base.flush(Base.stdout)
    status, (λ, P, warmstart) = open(solverlog, "a+") do logfile
        redirect_stdout(logfile) do
            status, (λ, P, warmstart) = PropertyT.solve(model, varλ, varP, warmstart)
            Base.Libc.flush_cstdio()
            status, (λ, P, warmstart)
        end
    end

    return status, (λ, P, warmstart)
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
    m.linconstrDuals = Array{Float64}(undef, 0)

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
                    @warn("Infeasibility ray (Farkas proof) not available")
                    fill(NaN, numRows)
                end
            elseif stat == :Unbounded
                m.colVal = try
                    unbdray = MathProgBase.getunboundedray(m.internalModel)
                    @assert length(unbdray) == numCols
                    unbdray
                catch
                    @warn("Unbounded ray not available")
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
        catch
            @warn("objBound could not be obtained")
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
        catch
            @warn("objVal/colVal could not be obtained")
        end
    end
    
    if traits.conic && m.objSense == :Max
        m.objBound = -1 * (m.objBound - m.obj.aff.constant) + m.obj.aff.constant
        m.objVal = -1 * (m.objVal - m.obj.aff.constant) + m.obj.aff.constant
    end

    return stat
end
