using JuMP


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

function create_sparse_product_matrix(matrix_constraints)
    row_indices = Int[]
    column_indices = Int[]
    values = Int[]
    for (index, pairs) in enumerate(matrix_constraints)
        for (i,j) in pairs
            push!(row_indices, i)
            push!(column_indices, j)
            push!(values, index)
        end
    end
    sparse_product_matrix = sparse(row_indices, column_indices, values)
    return sparse_product_matrix
end

function create_SDP_problem(matrix_constraints,
                            Δ²::GroupAlgebraElement, Δ::GroupAlgebraElement)
    N = length(Δ)
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
