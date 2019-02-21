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

function SOS_problem(X::GroupRingElem, orderunit::GroupRingElem; upper_bound::Float64=Inf)
    N = size(parent(X).pm, 1)
    m = JuMP.Model();

    JuMP.@variable(m, P[1:N, 1:N])
    JuMP.@constraint(m, P in PSDCone())
    JuMP.@constraint(m, sum(P[i] for i in eachindex(P)) == 0)

    if upper_bound < Inf
        λ = JuMP.@variable(m, λ <= upper_bound)
    else
        λ = JuMP.@variable(m, λ)
    end
    
    cnstrs = constraints(parent(X).pm)

    for (constraint, x, u) in zip(cnstrs, X.coeffs, orderunit.coeffs)
        JuMP.@constraint(m, sum(P[constraint]) == x - λ*u)
    end
    
    JuMP.@objective(m, Max, λ)
    
    return m
end

###############################################################################
#
#  Symmetrized SDP
#
###############################################################################

function SOS_problem(X::GroupRingElem, orderunit::GroupRingElem, data::OrbitData; upper_bound::Float64=Inf)
    Ns = size.(data.Uπs, 2)
    m = JuMP.Model();

    Ps = Vector{Matrix{JuMP.VariableRef}}(undef, length(Ns))

    for (k,s) in enumerate(Ns)
        Ps[k] = JuMP.@variable(m, [1:s, 1:s])
        JuMP.@constraint(m, Ps[k] in PSDCone())
    end

    if upper_bound < Inf
        λ = JuMP.@variable(m, λ <= upper_bound)
    else
        λ = JuMP.@variable(m, λ)
    end
    
    @info("Adding $(length(data.orbits)) constraints... ")
    @time addconstraints!(m, Ps, X, orderunit, data)

    JuMP.@objective(m, Max, λ)
        
    return m, Ps
end

function constraintLHS!(M, cnstr, Us, Ust, dims, eps=1000*eps(eltype(first(M))))
    for π in 1:lastindex(Us)
        M[π] = dims[π].*PropertyT.clamp_small!(Ust[π]*cnstr*Us[π], eps)
    end
end

function addconstraints!(m::JuMP.Model,
    P::Vector{Matrix{JuMP.VariableRef}},
    X::GroupRingElem, orderunit::GroupRingElem, data::OrbitData)

    orderunit_orb = orbit_spvector(orderunit.coeffs, data.orbits)
    X_orb = orbit_spvector(X.coeffs, data.orbits)
    UπsT = [U' for U in data.Uπs]

    cnstrs = constraints(parent(X).pm)
    orb_cnstr = spzeros(Float64, size(parent(X).pm)...)

    M = [Array{Float64}(undef, n,n) for n in size.(UπsT,1)]
    
    λ = m[:λ]

    for (t, orbit) in enumerate(data.orbits)
        orbit_constraint!(orb_cnstr, cnstrs, orbit)
        constraintLHS!(M, orb_cnstr, data.Uπs, UπsT, data.dims)
        
        x, u = X_orb[t], orderunit_orb[t]
        
        @constraints m begin
            x - λ*u == sum(dot(M[π], P[π]) for π in eachindex(data.Uπs))
        end
    end
    return m
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


function solve(m::JuMP.Model, with_optimizer::JuMP.OptimizerFactory, warmstart=nothing)
    
    set_optimizer(m, with_optimizer)
    MOIU.attach_optimizer(m)
    
    if warmstart != nothing
        p_sol, d_sol, s = warmstart
        MathProgBase.SolverInterface.setwarmstart!(m.internalModel, p_sol;
            dual_sol=d_sol, slack=s);
    end

    optimize!(m)
    status = termination_status(m)

    return status, (λ, P, warmstart)
end

function solve(solverlog::String, m::JuMP.Model, with_optimizer::JuMP.OptimizerFactory, warmstart=nothing)

    isdir(dirname(solverlog)) || mkpath(dirname(solverlog))

    Base.flush(Base.stdout)
    status, warmstart = open(solverlog, "a+") do logfile
        redirect_stdout(logfile) do
            status, warmstart = PropertyT.solve(m, with_optimizer, warmstart)
            Base.Libc.flush_cstdio()
            status, warmstart
        end
    end

    return status, warmstart
end
