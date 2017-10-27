using JuMP
using SCS

export Settings, OrbitData

immutable Settings
   name::String
   N::Int
   G::Group
   S::Vector
   AutS::Group
   radius::Int
   solver::AbstractMathProgSolver
   upper_bound::Float64
   tol::Float64
end

prefix(s::Settings) = s.name
suffix(s::Settings) = "$(s.upper_bound)"
prepath(s::Settings) = prefix(s)
fullpath(s::Settings) = joinpath(prefix(s), suffix(s))

immutable OrbitData
   name::String
   Us::Vector
   Ps::Vector{Array{JuMP.Variable,2}}
   cnstr::Vector
   laplacian::Vector
   laplacianSq::Vector
   dims::Vector{Int}
end

function OrbitData(sett::Settings)
   splap = load(joinpath(prepath(sett), "delta.jld"), "Δ");
   pm = load(joinpath(prepath(sett), "pm.jld"), "pm");
   cnstr = PropertyT.constraints_from_pm(pm);
   splap² = similar(splap)
   splap² = GroupRings.mul!(splap², splap, splap, pm);

   # Uπs = load(joinpath(name, "U_pis.jld"), "Uπs");
   Uπs = load(joinpath(prepath(sett), "U_pis.jld"), "spUπs");
   #dimensions of the corresponding πs:
   dims = load(joinpath(prepath(sett), "U_pis.jld"), "dims")

   m, P = init_model(Uπs);

   orbits = load(joinpath(prepath(sett), "orbits.jld"), "orbits");
   n = size(Uπs[1],1)
   orb_spcnstrm = [orbit_constraint(cnstr[collect(orb)], n) for orb in orbits]
   orb_splap = orbit_spvector(splap, orbits)
   orb_splap² = orbit_spvector(splap², orbits)

   orbData = OrbitData(fullpath(sett), Uπs, P, orb_spcnstrm, orb_splap, orb_splap², dims);

   # orbData = OrbitData(name, Uπs, P, orb_spcnstrm, splap, splap², dims);

   return m, orbData
end

include("OrbitDecomposition.jl")

dens(M::SparseMatrixCSC) = length(M.nzval)/length(M)
dens(M::AbstractArray) = length(findn(M)[1])/length(M)

function sparsify!{Tv,Ti}(M::SparseMatrixCSC{Tv,Ti}, eps=eps(Tv); verbose=false)
   n = nnz(M)

   densM = dens(M)
   for i in eachindex(M.nzval)
      if abs(M.nzval[i]) < eps
         M.nzval[i] = zero(Tv)
      end
   end
   dropzeros!(M)
   m = nnz(M)

   if verbose
      info(logger, "Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M), 20))
   end

   return M
end

function sparsify!{T}(M::AbstractArray{T}, eps=eps(T); check=false, verbose=false)
   densM = dens(M)
   rankM = rank(M)
   M[abs.(M) .< eps] .= zero(T)

   if check && rankM != rank(M)
      warn(logger, "Sparsification decreased the rank!")
   end

   if verbose
      info(logger, "Sparsified density:", rpad(densM, 20), " → ", rpad(dens(M),20))
   end

   return M
end

sparsify{T}(U::AbstractArray{T}, tol=eps(T)) = sparsify!(deepcopy(U), tol)

function init_orbit_data(logger, sett::Settings; radius=2)

   ex(fname) = isfile(joinpath(prepath(sett), fname))

   files_exists = ex.(["delta.jld", "pm.jld", "U_pis.jld", "orbits.jld"])

   if !all(files_exists)
      compute_orbit_data(logger, prepath(sett), sett.G, sett.S, sett.AutS, radius=radius)
   end

   return 0
end

function transform(U::AbstractArray, V::AbstractArray; sparse=true)
   if sparse
      return sparsify!(U'*V*U)
   else
      return U'*V*U
   end
end

A(data::OrbitData, π, t) = data.dims[π].*transform(data.Us[π], data.cnstr[t])

function constrLHS(m::JuMP.Model, data::OrbitData, t)
    l = endof(data.Us)
    lhs = @expression(m, sum(vecdot(A(data, π, t), data.Ps[π]) for π in 1:l))
    return lhs
end

function constrLHS(m::JuMP.Model, cnstr, Us, Ust, dims, vars, eps=100*eps(1.0))
   M = [PropertyT.sparsify!(dims[π].*Ust[π]*cnstr*Us[π], eps) for π in 1:endof(Us)]
   return @expression(m, sum(vecdot(M[π], vars[π]) for π in 1:endof(Us)))
end

function addconstraints!(m::JuMP.Model, data::OrbitData, l::Int=length(data.laplacian); var::Symbol=:λ)
   λ = m[var]
   Ust = [U' for U in data.Us]
   idx = [π for π in 1:endof(data.Us) if size(data.Us[π],2) != 0]

   for t in 1:l
      if t % 100 == 0
         print(t, ", ")
      end
   #   lhs = constrLHS(m, data, t)
      lhs = constrLHS(m, data.cnstr[t], data.Us[idx], Ust[idx], data.dims[idx], data.Ps[idx])

      d, d² = data.laplacian[t], data.laplacianSq[t]
      # if lhs == zero(lhs)
      #    if d == 0 && d² == 0
      #        info("Detected empty constraint")
      #        continue
      #    else
      #        warn("Adding unsatisfiable constraint!")
      #    end
      # end
      JuMP.@constraint(m, lhs == d² - λ*d)
   end
   println("")
end

function init_model(Uπs)
    m = JuMP.Model();
    l = size(Uπs,1)
    P = Vector{Array{JuMP.Variable,2}}(l)

    for k in 1:l
        s = size(Uπs[k],2)
        if s > 0
           P[k] = JuMP.@variable(m, [i=1:s, j=1:s])
           JuMP.@SDconstraint(m, P[k] >= 0.0)
        end
    end

    JuMP.@variable(m, λ >= 0.0)
    JuMP.@objective(m, Max, λ)
    return m, P
end

function create_SDP_problem(sett::Settings)
   info(logger, "Loading orbit data....")
   t = @timed SDP_problem, orb_data = OrbitData(sett);
   info(logger, PropertyT.timed_msg(t))

   if sett.upper_bound < Inf
      λ = JuMP.getvariable(SDP_problem, :λ)
      JuMP.@constraint(SDP_problem, λ <= sett.upper_bound)
   end

   t = length(orb_data.laplacian)
   info(logger, "Adding $t constraints ... ")
   t = @timed addconstraints!(SDP_problem, orb_data)
   info(logger, PropertyT.timed_msg(t))

   return SDP_problem, orb_data
end

function λandP(m::JuMP.Model, data::OrbitData)
   varλ = m[:λ]
   varP = data.Ps
   λ, Ps = PropertyT.λandP(data.name, m, varλ, varP)
   return λ, Ps
end

function λandP(m::JuMP.Model, data::OrbitData, sett::Settings)
   info(logger, "Solving SDP problem...")
   λ, Ps = λandP(m, data)

   info(logger, "Reconstructing P...")

   t = @timed preps = perm_reps(sett.G, sett.S, sett.AutS, sett.radius)
   info(logger, PropertyT.timed_msg(t))

   t = @timed recP = reconstruct_sol(preps, data.Us, Ps, data.dims)
   info(logger, PropertyT.timed_msg(t))

   fname = PropertyT.λSDPfilenames(fullpath(sett))[2]
   save(fname, "origP", Ps, "P", recP)
   return λ, recP
end

function check_property_T(sett::Settings)

   init_orbit_data(logger, sett, radius=sett.radius)

   fnames = PropertyT.λSDPfilenames(fullpath(sett))

   if all(isfile.(fnames))
      λ, P = PropertyT.λandP(fullpath(sett))
   else
      info(logger, "Creating SDP problem...")
      SDP_problem, orb_data = create_SDP_problem(sett)
      JuMP.setsolver(SDP_problem, sett.solver)

      λ, P = λandP(SDP_problem, orb_data, sett)
   end

   info(logger, "λ = $λ")
   info(logger, "sum(P) = $(sum(P))")
   info(logger, "maximum(P) = $(maximum(P))")
   info(logger, "minimum(P) = $(minimum(P))")

   if λ > 0
      pm_fname = joinpath(prepath(sett), "pm.jld")
      RG = GroupRing(sett.G, load(pm_fname, "pm"))
      Δ_fname = joinpath(prepath(sett), "delta.jld")
      Δ = GroupRingElem(load(Δ_fname, "Δ")[:, 1], RG)

      isapprox(eigvals(P), abs.(eigvals(P)), atol=sett.tol) ||
          warn("The solution matrix doesn't seem to be positive definite!")
     #  @assert P == Symmetric(P)
      Q = real(sqrtm(Symmetric(P)))

      sgap = PropertyT.check_distance_to_positive_cone(Δ, λ, Q, 2*sett.radius, tol=sett.tol, rational=false)
      if isa(sgap, Interval)
           sgap = sgap.lo
      end
      if sgap > 0
           info(logger, "λ ≥ $(Float64(trunc(sgap,12)))")
            Kazhdan_κ = PropertyT.Kazhdan_from_sgap(sgap, length(sett.S))
            Kazhdan_κ = Float64(trunc(Kazhdan_κ, 12))
            info(logger, "κ($(sett.name), S) ≥ $Kazhdan_κ: Group HAS property (T)!")
            return true
      else
           sgap = Float64(trunc(sgap, 12))
           info(logger, "λ($(sett.name), S) ≥ $sgap: Group may NOT HAVE property (T)!")
           return false
      end
   end
   info(logger, "κ($(sett.name), S) ≥ $λ < 0: Tells us nothing about property (T)")
   return false
end
