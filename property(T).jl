using JuMP

function products{T<:Real}(S1::Array{Array{T,2},1}, S2::Array{Array{T,2},1})
    result = [0*similar(S1[1])]
    for x in S1
        for y in S2
            push!(result, x*y)
        end
    end
    return unique(result[2:end])
end

function read_GAP_raw_list(filename::String)
    return eval(parse(String(read(filename))))
end

function create_product_matrix(matrix_constraints)
    l = length(matrix_constraints)
    product_matrix = zeros(Int, (l, l))
    for (index, pairs) in enumerate(matrix_constraints)
        for (i,j) in pairs
            product_matrix[i,j] = index
        end
    end
    return product_matrix
end

function create_product_matrix(basis::Array{Array{Float64,2},1}, limit::Int)

    product_matrix = Array{Int}(limit,limit)
    constraints = [Array{Int,1}[] for x in 1:length(basis)]

    for i in 1:limit
        x_inv = inv(basis[i])
        for j in 1:limit
            w::Array{Float64,2} = x_inv*basis[j]

            function f(x::Array{Float64,2})
                if x == w
                    return true
                else
                    return false
                end
            end
            index = findfirst(f, basis)
            product_matrix[i,j] = index
            push!(constraints[index],[i,j])
        end
    end
    return product_matrix, constraints
end

function Laplacian_sparse(S::Array{Array{Float64,2},1},
    basis::Array{Array{Float64,2},1})

    squares = unique(vcat([basis[1]], S, products(S,S)))

    result = spzeros(length(basis))
    result[1] = length(S)
    for s in S
        ind = find(x -> x==s, basis)
        result[ind] += -1
    end
    return result
end

function Laplacian(S::Array{Array{Float64,2},1},
    basis:: Array{Array{Float64,2},1})

    return full(Laplacian_sparse(S,basis))
end

function create_SDP_problem(matrix_constraints,
                            Δ²::GroupAlgebraElement, Δ::GroupAlgebraElement)
    N = size(Δ.product_matrix,1)
    @assert length(Δ) == length(Δ²)
    @assert length(Δ) == length(matrix_constraints)
    m = Model();
    @variable(m, A[1:N, 1:N], SDP)
    @SDconstraint(m, A >= zeros(size(A)))
    @variable(m, κ >= 0.0)
    @objective(m, Max, κ)

    for (pairs, δ², δ) in zip(matrix_constraints, Δ².coordinates, Δ.coordinates)
        @constraint(m, sum(A[i,j] for (i,j) in pairs) == δ² - κ*δ)
    end
    return m
end

function resulting_SOS{T<:Number, S<:Number}(sqrt_matrix::Array{T,2}, elt::GroupAlgebraElement{S})
    zzz = zeros(T, size(sqrt_matrix)[1])
    result::GroupAlgebraElement{T} = GroupAlgebraElement(zzz, elt.product_matrix)
    for i in 1:length(result)
        new_base = GroupAlgebraElement(sqrt_matrix[:,i], elt.product_matrix)
        result += new_base*new_base
    end
    return result
end

function correct_to_augmentation_ideal{T<:Rational}(sqrt_matrix::Array{T,2})
    sqrt_corrected = similar(sqrt_matrix)
    l = size(sqrt_matrix,2)
    for i in 1:l
        col = view(sqrt_matrix,:,i)
        sqrt_corrected[:,i] = col - sum(col)//l
#         @assert sum(sqrt_corrected[:,i]) == 0
    end
    return sqrt_corrected
end
