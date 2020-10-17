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

function SOS_problem_dual(elt::GroupRingElem, order_unit::GroupRingElem;
    lower_bound::Float64=Inf)
    @assert parent(elt) == parent(order_unit)

    RG = parent(elt)
    m = Model()

    y = @variable(m, y[1:length(elt.coeffs)])

    @constraint(m, λ_dual, dot(order_unit.coeffs, y) == 1)
    @constraint(m, psd, [y[i] for i in RG.pm] in PSDCone())

    if !isinf(lower_bound)
        @variable(m, λ_ub_dual)
        expr = dot(elt.coeffs, y) + lower_bound*λ_ub_dual
        # @constraint m expr >= lower_bound
        @objective m Min expr
    else
        @objective m Min dot(elt.coeffs, y)
    end

    return m
end

function SOS_problem_primal(X::GroupRingElem, orderunit::GroupRingElem;
    upper_bound::Float64=Inf)

    N = size(parent(X).pm, 1)
    m = JuMP.Model();

    JuMP.@variable(m, P[1:N, 1:N])
    # SP = Symmetric(P)
    JuMP.@SDconstraint(m, sdp, P >= 0)

    if iszero(aug(X)) && iszero(aug(orderunit))
        JuMP.@constraint(m, augmentation, sum(P) == 0)
    end

    if upper_bound < Inf
        λ = JuMP.@variable(m, λ <= upper_bound)
    else
        λ = JuMP.@variable(m, λ)
    end

    cnstrs = constraints(parent(X).pm)
    @assert length(cnstrs) == length(X.coeffs) == length(orderunit.coeffs)

    x, u = X.coeffs, orderunit.coeffs
    JuMP.@constraint(m, lincnstr[i=1:length(cnstrs)],
        x[i] - λ*u[i] == sum(P[cnstrs[i]]))

    JuMP.@objective(m, Max, λ)

    return m
end

###############################################################################
#
#  Symmetrized SDP
#
###############################################################################

function SOS_problem_primal(X::GroupRingElem, orderunit::GroupRingElem, data::OrbitData; upper_bound::Float64=Inf)
    Ns = size.(data.Uπs, 2)
    m = JuMP.Model();

    Ps = Vector{Matrix{JuMP.VariableRef}}(undef, length(Ns))

    for (k,s) in enumerate(Ns)
        Ps[k] = JuMP.@variable(m, [1:s, 1:s])
        JuMP.@SDconstraint(m, Ps[k] >= 0)
    end

    if upper_bound < Inf
        λ = JuMP.@variable(m, λ <= upper_bound)
    else
        λ = JuMP.@variable(m, λ)
    end

    @info "Adding $(length(data.orbits)) constraints..."
    @time addconstraints!(m, Ps, X, orderunit, data)

    JuMP.@objective(m, Max, λ)

    return m, Ps
end

function constraintLHS!(M, cnstr, Us, Ust, dims, eps=1000*eps(eltype(first(M))))
    for π in 1:lastindex(Us)
        M[π] = dims[π].*PropertyT.clamp_small!(Ust[π]*(cnstr*Us[π]), eps)
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

        JuMP.@constraints m begin
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
        {M<:AbstractMatrix, GEl<:GroupElem, P<:Generic.Perm, U<:AbstractMatrix}

    lU = length(Uπs)
    transfP = [dims[π].*Uπs[π]*Ps[π]*Uπs[π]' for π in 1:lU]
    tmp = [zeros(Float64, size(first(transfP))) for _ in 1:lU]

    Threads.@threads for π in 1:lU
        tmp[π] = perm_avg!(tmp[π], transfP[π], values(preps))
    end

    recP = sum(tmp)./length(preps)

    return recP
end

function perm_avg!(result, P, perms)
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

function setwarmstart!(m::JuMP.Model, warmstart)
    if solver_name(m) == "SCS"
        primal, dual, slack = warmstart
        m.moi_backend.optimizer.model.optimizer.data.primal = primal
        m.moi_backend.optimizer.model.optimizer.data.dual = dual
        m.moi_backend.optimizer.model.optimizer.data.slack = slack
    else
        @warn "Setting warmstart for $(solver_name(m)) is not implemented! Ignoring..."
    end
    return m
end

function getwarmstart(m::JuMP.Model)
    if solver_name(m) == "SCS"
        warmstart = (
            primal = m.moi_backend.optimizer.model.optimizer.data.primal,
            dual = m.moi_backend.optimizer.model.optimizer.data.dual,
            slack = m.moi_backend.optimizer.model.optimizer.data.slack
            )
    else
        @warn "Saving warmstart for $(solver_name(m)) is not implemented!"
        return (primal=Float64[], dual=Float64[], slack=Float64[])
    end
    return warmstart
end

function solve(m::JuMP.Model, with_optimizer::JuMP.OptimizerFactory, warmstart=nothing)

    set_optimizer(m, with_optimizer)
    MOIU.attach_optimizer(m)

    if warmstart != nothing
        setwarmstart!(m, warmstart)
    end

    optimize!(m)
    Base.Libc.flush_cstdio()

    status = termination_status(m)

    return status, getwarmstart(m)
end

function solve(solverlog::String, m::JuMP.Model, with_optimizer::JuMP.OptimizerFactory, warmstart=nothing)

    isdir(dirname(solverlog)) || mkpath(dirname(solverlog))

    Base.flush(Base.stdout)
    status, warmstart = open(solverlog, "a+") do logfile
        redirect_stdout(logfile) do
            status, warmstart = PropertyT.solve(m, with_optimizer, warmstart)
            status, warmstart
        end
    end

    return status, warmstart
end
