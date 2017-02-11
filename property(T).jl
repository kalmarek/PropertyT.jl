using JuMP
import Base: rationalize
using GroupAlgebras

function products(S1, S2)
    result = Vector{eltype(S1[1]*S2[1])}()
    for x in S1
        for y in S2
            push!(result, x*y)
        end
    end
    return unique(result)
end

function generate_B₂_and_B₄(identity, S₁)
    S₂ = products(S₁, S₁);
    S₃ = products(S₁, S₂);
    S₄ = products(S₂, S₂);

    B₂ = unique(vcat([identity],S₁,S₂));
    B₄ = unique(vcat(B₂, S₃, S₄));
    @assert B₄[1:length(B₂)] == B₂
    return B₂, B₄;
end

function read_GAP_raw_list(filename)
    return eval(parse(String(read(filename))))
end


function create_product_matrix(basis::Array{Array{Float64,2},1}, limit)

    product_matrix = zeros(Int, (limit,limit))
    constraints = [Array{Int,1}[] for x in 1:length(basis)]

    for i in 1:limit
        x_inv = inv(basis[i])
        # info("$i of $limit")
        for j in 1:limit
            w = x_inv*basis[j]

            index = findfirst(basis, w)
            if 0 < index ≤ limit
                product_matrix[i,j] = index
                push!(constraints[index],[i,j])
            end
        end
    end
    return product_matrix, constraints
end

function constraints_from_pm(pm, total_length=maximum(pm))
    n = size(pm,1)
    constraints = constraints = [Array{Int,1}[] for x in 1:total_length]
    for j in 1:n
        Threads.@threads for i in 1:n
            idx = pm[i,j]
            push!(constraints[idx], [i,j])
        end
    end
    return constraints
end
function Laplacian_sparse(S::Array{Array{Float64,2},1},
    basis::Array{Array{Float64,2},1})

    squares = unique(vcat([basis[1]], S, products(S,S)))

    result = spzeros(length(basis))
    result[1] = length(S)
    for s in S
        ind = find(basis, s)
        result[ind] += -1
    end
    return result
end

function Laplacian(S::Array{Array{Float64,2},1},
    basis:: Array{Array{Float64,2},1})

    return full(Laplacian_sparse(S,basis))
end

function prepare_Laplacian_and_constraints{T}(S::Vector{Array{T,2}})

    identity = eye(S[1])
    B₂, B₄ = generate_B₂_and_B₄(identity, S)
    product_matrix, matrix_constraints = create_product_matrix(B₄,length(B₂));

    L= Laplacian(S, B₄);

    return GroupAlgebraElement(L, product_matrix), matrix_constraints
end

function create_SDP_problem(matrix_constraints, Δ::GroupAlgebraElement)
    N = size(Δ.product_matrix,1)
    const Δ² = Δ*Δ
    @assert length(Δ) == length(matrix_constraints)
    m = Model();
    @variable(m, A[1:N, 1:N], SDP)
    @SDconstraint(m, A >= zeros(size(A)))
    @variable(m, κ >= 0.0)
    @objective(m, Max, κ)

    for (pairs, δ², δ) in zip(matrix_constraints, Δ².coefficients, Δ.coefficients)
        @constraint(m, sum(A[i,j] for (i,j) in pairs) == δ² - κ*δ)
    end
    return m
end

function solve_for_property_T{T}(S₁::Vector{Array{T,2}}, solver; verbose=true)

    Δ, matrix_constraints = prepare_Laplacian_and_constraints(S₁)

    problem = create_SDP_problem(matrix_constraints, Δ);
    @show solver

    setsolver(problem, solver);
    verbose && @show problem

    solution_status = solve(problem);
    verbose && @show solution_status

    if solution_status != :Optimal
        throw(ExceptionError("The solver did not solve the problem successfully!"))
    else
        κ = SL_3ZZ.objVal;
        A = getvalue(getvariable(SL_3ZZ, :A));;
    end

    return κ, A
end

function EOI{T<:Number}(Δ::GroupAlgebraElement{T}, κ::T)
    return Δ*Δ - κ*Δ
end

@everywhere function square(vector, elt)
    zzz = zeros(elt.coefficients)
    zzz[1:length(vector)] = vector
#     new_base_elt = GroupAlgebraElement(zzz, elt.product_matrix)
#     return (new_base_elt*new_base_elt).coefficients
    return GroupAlgebras.algebra_multiplication(zzz, zzz, elt.product_matrix)
end

function compute_SOS{T<:Number}(sqrt_matrix::Array{T,2},
                                  elt::GroupAlgebraElement{T})
    L = size(sqrt_matrix,2)
    result = @parallel (+) for i in 1:L
        square(sqrt_matrix[:,i], elt)
    end
    return GroupAlgebraElement{T}(result, elt.product_matrix)
end

function correct_to_augmentation_ideal{T<:Rational}(sqrt_matrix::Array{T,2})
    sqrt_corrected = similar(sqrt_matrix)
    l = size(sqrt_matrix,2)
    for i in 1:l
        col = view(sqrt_matrix,:,i)
        sqrt_corrected[:,i] = col - sum(col)//l
        # @assert sum(sqrt_corrected[:,i]) == 0
    end
    return sqrt_corrected
end

function check_solution{T<:Number}(κ::T,
                                   sqrt_matrix::Array{T,2},
                                   Δ::GroupAlgebraElement{T})
    eoi = EOI(Δ, κ)
    result = compute_SOS(sqrt_matrix, Δ)
    L₁_dist = norm(result - eoi,1)
    return eoi - result, L₁_dist
end

function rationalize{T<:Integer, S<:Real}(::Type{T},
    X::AbstractArray{S}; tol::Real=eps(eltype(X)))
    r(x) = rationalize(T, x, tol=tol)
    return r.(X)
end;
