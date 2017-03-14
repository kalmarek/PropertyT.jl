using JuMP
import MathProgBase: AbstractMathProgSolver

function create_product_matrix(basis, limit)
    product_matrix = zeros(Int, (limit,limit))
    basis_dict = Dict{Array, Int}(x => i
        for (i,x) in enumerate(basis))
    for i in 1:limit
        x_inv::eltype(basis) = inv(basis[i])
        for j in 1:limit
            w = x_inv*basis[j]
            product_matrix[i,j] = basis_dict[w]
            # index = findfirst(basis, w)
            # index ≠ 0 || throw(ArgumentError("Product is not supported on basis: $w"))
            # product_matrix[i,j] = index
        end
    end
    return product_matrix
end

function constraints_from_pm(pm, total_length)
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

constraints_from_pm(pm) = constraints_from_pm(pm, maximum(pm))

function splaplacian_coeff(S, basis, n=length(basis))
    result = spzeros(n)
    result[1] = float(length(S))
    for s in S
        ind = findfirst(basis, s)
        result[ind] += -1.0
    end
    return result
end

function laplacian_coeff(S, basis)
    return full(splaplacian_coeff(S,basis))
end


function create_SDP_problem(matrix_constraints, Δ::GroupAlgebraElement; upper_bound=Inf)
    N = size(Δ.product_matrix,1)
    const Δ² = Δ*Δ
    @assert length(Δ) == length(matrix_constraints)
    m = JuMP.Model();
    JuMP.@variable(m, A[1:N, 1:N], SDP)
    JuMP.@SDconstraint(m, A >= 0)
    JuMP.@constraint(m, sum(A[i] for i in eachindex(A)) == 0)
    JuMP.@variable(m, κ >= 0.0)
    if upper_bound < Inf
        JuMP.@constraint(m, κ <= upper_bound)
    end
    JuMP.@objective(m, Max, κ)

    for (pairs, δ², δ) in zip(matrix_constraints, Δ².coefficients, Δ.coefficients)
        JuMP.@constraint(m, sum(A[i,j] for (i,j) in pairs) == δ² - κ*δ)
    end
    return m
end

function solve_SDP(SDP_problem, solver)
    @show SDP_problem
    @show solver

    JuMP.setsolver(SDP_problem, solver);
    # @time MathProgBase.writeproblem(SDP_problem, "/tmp/SDP_problem")
    solution_status = JuMP.solve(SDP_problem);

    if solution_status != :Optimal
        warn("The solver did not solve the problem successfully!")
    end
    @show solution_status

    κ = JuMP.getvalue(JuMP.getvariable(SDP_problem, :κ))
    A = JuMP.getvalue(JuMP.getvariable(SDP_problem, :A))
    return κ, A
end
