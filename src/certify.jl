function augment_columns!(Q::AbstractMatrix)
    for c in eachcol(Q)
        c .-= sum(c) ./ length(c)
    end
    return Q
end

function _fma_SOS_thr!(
    result::AbstractVector{T},
    mstructure::AbstractMatrix{<:Integer},
    Q::AbstractMatrix{T},
    acc_matrix=zeros(T, size(mstructure)...),
) where {T}

    s1, s2 = size(mstructure)

    @inbounds for k = 1:s2
        let k = k, s1 = s1, s2 = s2, Q = Q, acc_matrix = acc_matrix
            Threads.@threads for j = 1:s2
                for i = 1:s1
                    @inbounds acc_matrix[i, j] =
                        muladd(Q[i, k], Q[j, k], acc_matrix[i, j])
                end
            end
        end
    end

    @inbounds for j = 1:s2
        for i = 1:s1
            result[mstructure[i, j]] += acc_matrix[i, j]
        end
    end

    return result
end

function _cnstr_sos!(res::AlgebraElement, Q::AbstractMatrix, cnstrs)
    StarAlgebras.zero!(res)
    Q² = Q' * Q
    for (g, A_g) in cnstrs
        res[g] = dot(A_g, Q²)
    end
    return res
end

function _augmented_sos!(res::AlgebraElement, Q::AbstractMatrix)
    A = parent(res)
    StarAlgebras.zero!(res)
    Q² = Q' * Q

    N = LinearAlgebra.checksquare(A.mstructure)
    augmented_basis = [A(1) - A(b) for b in @view basis(A)[1:N]]
    tmp = zero(res)

    for (j, y) in enumerate(augmented_basis)
        for (i, x) in enumerate(augmented_basis)
            # res += Q²[i, j] * x * y

            StarAlgebras.mul!(tmp, x, y)
            StarAlgebras.mul!(tmp, tmp, Q²[i, j])
            StarAlgebras.add!(res, res, tmp)
        end
    end
    return res
end

function compute_sos(A::StarAlgebra, Q::AbstractMatrix; augmented::Bool)
    if augmented
        z = zeros(eltype(Q), length(basis(A)))
        res = AlgebraElement(z, A)
        return _augmented_sos!(res, Q)
        cnstrs = constraints(basis(A), A.mstructure; augmented=true)
        return _cnstr_sos!(res, Q, cnstrs)
    else
        @assert size(A.mstructure) == size(Q)
        z = zeros(eltype(Q), length(basis(A)))

        _fma_SOS_thr!(z, A.mstructure, Q)

        return AlgebraElement(z, A)
    end
end

function sufficient_λ(residual::AlgebraElement, λ; halfradius)
    L1_norm = norm(residual, 1)
    suff_λ = λ - 2.0^(2ceil(log2(halfradius))) * L1_norm

    eq_sign = let T = eltype(residual)
        if T <: Interval
            "∈"
        elseif T <: Union{Rational,Integer}
            "="
        else # if T <: AbstractFloat
            "≈"
        end
    end

    info_strs = [
        "Numerical metrics of the obtained SOS:",
        "ɛ(elt - λu - ∑ξᵢ*ξᵢ) $eq_sign $(aug(residual))",
        "‖elt - λu - ∑ξᵢ*ξᵢ‖₁ $eq_sign $(L1_norm)",
        " λ $eq_sign $suff_λ",
    ]
    @info join(info_strs, "\n")

    return suff_λ
end

function sufficient_λ(
    elt::AlgebraElement,
    order_unit::AlgebraElement,
    λ,
    sos::AlgebraElement;
    halfradius
)

    @assert parent(elt) === parent(order_unit) == parent(sos)
    residual = (elt - λ * order_unit) - sos

    return sufficient_λ(residual, λ; halfradius=halfradius)
end

function certify_solution(
    elt::AlgebraElement,
    orderunit::AlgebraElement,
    λ,
    Q::AbstractMatrix{<:AbstractFloat};
    halfradius,
    augmented=iszero(aug(elt)) && iszero(aug(orderunit))
)

    should_we_augment = !augmented && aug(elt) == aug(orderunit) == 0

    Q = should_we_augment ? augment_columns!(Q) : Q
    @time sos = compute_sos(parent(elt), Q, augmented=augmented)

    @info "Checking in $(eltype(sos)) arithmetic with" λ

    λ_flpoint = sufficient_λ(elt, orderunit, λ, sos, halfradius=halfradius)

    if λ_flpoint ≤ 0
        return false, λ_flpoint
    end

    λ_int = @interval(λ)
    Q_int = [@interval(q) for q in Q]

    check, sos_int = @time if should_we_augment
        @info("Projecting columns of Q to the augmentation ideal...")
        Q_int = augment_columns!(Q_int)
        @info "Checking that sum of every column contains 0.0..."
        check_augmented = all(0 ∈ sum(c) for c in eachcol(Q_int))
        check_augmented || @error(
            "Augmentation failed! The following numbers are not certified!"
        )
        sos_int = compute_sos(parent(elt), Q_int; augmented=augmented)
        check_augmented, sos_int
    else
        true, compute_sos(parent(elt), Q_int, augmented=augmented)
    end

    @info "Checking in $(eltype(sos_int)) arithmetic with" λ

    λ_certified =
        sufficient_λ(elt, orderunit, λ_int, sos_int, halfradius=halfradius)

    return check && inf(λ_certified) > 0.0, inf(λ_certified)
end
