import IntervalArithmetic
import IntervalMatrices

function augment_columns!(Q::AbstractMatrix)
    for c in eachcol(Q)
        c .-= sum(c) ./ length(c)
    end
    return Q
end

function __sos_via_square!(
    res::SA.AlgebraElement,
    P::AbstractMatrix;
    augmented::Bool,
    id = (b = SA.basis(parent(res)); b[one(first(b))]),
)
    A = parent(res)
    mstr = A.mstructure
    @assert size(mstr) == size(P)

    SA.zero!(res)
    for j in axes(mstr, 2)
        for i in axes(mstr, 1)
            p = P[i, j]
            x_star_y = mstr[-i, j]
            res[x_star_y] += p
            # either result += P[x,y]*(x'*y)
            if augmented
                # or result += P[x,y]*(1-x)'*(1-y) == P[x,y]*(1-x'-y+x'y)
                res[id] += p
                x_star, y = mstr[-i, id], j
                res[x_star] -= p
                res[y] -= p
            end
        end
    end

    return res
end

function __sos_via_cnstr!(
    res::SA.AlgebraElement,
    Q²::AbstractMatrix,
    cnstrs,
)
    SA.zero!(res)
    for (g, A_g) in cnstrs
        res[g] = dot(A_g, Q²)
    end
    return res
end

__square(Q::AbstractMatrix) = Q' * Q
function __square(Q::IntervalMatrices.IntervalMatrix)
    return *(IntervalMatrices.MultiplicationType{:fast}(), Q', Q)
end

function compute_sos(
    A::SA.StarAlgebra,
    Q::AbstractMatrix;
    augmented::Bool,
)
    Q² = __square(Q)
    res = SA.AlgebraElement(zeros(eltype(Q²), length(SA.basis(A))), A)
    res = __sos_via_square!(res, Q²; augmented = augmented)
    return res
end

function sufficient_λ(residual::SA.AlgebraElement, λ; halfradius)
    L1_norm = norm(residual, 1)
    suff_λ = λ - 2.0^(2ceil(log2(halfradius))) * L1_norm

    eq_sign = let T = eltype(residual)
        if T <: IntervalArithmetic.Interval
            "∈"
        elseif T <: Union{Rational,Integer}
            "="
        else # if T <: AbstractFloat
            "≈"
        end
    end

    info_strs = [
        "Numerical metrics of the obtained SOS:",
        "ɛ(elt - λu - ∑ξᵢ*ξᵢ) $eq_sign " *
        sprint(show, SA.aug(residual); context = :compact => true),
        "‖elt - λu - ∑ξᵢ*ξᵢ‖₁ $eq_sign " *
        sprint(show, L1_norm; context = :compact => true),
        " λ $eq_sign $suff_λ",
    ]
    @info join(info_strs, "\n")

    return suff_λ
end

function sufficient_λ(
    elt::SA.AlgebraElement,
    order_unit::SA.AlgebraElement,
    λ,
    sos::SA.AlgebraElement;
    halfradius,
)
    @assert parent(elt) === parent(order_unit) == parent(sos)
    residual = (elt - λ * order_unit) - sos

    return sufficient_λ(residual, λ; halfradius = halfradius)
end

function certify_solution(
    elt::SA.AlgebraElement,
    orderunit::SA.AlgebraElement,
    λ,
    Q::AbstractMatrix{<:AbstractFloat};
    halfradius,
    augmented = iszero(SA.aug(elt)) && iszero(SA.aug(orderunit)),
)
    should_we_augment =
        !augmented && SA.aug(elt) == SA.aug(orderunit) == 0

    Q = should_we_augment ? augment_columns!(Q) : Q
    @info "Checking in $(eltype(Q)) arithmetic with" λ
    @time sos = compute_sos(parent(elt), Q; augmented = augmented)

    λ_flpoint = sufficient_λ(elt, orderunit, λ, sos; halfradius = halfradius)

    if λ_flpoint ≤ 0
        return false, λ_flpoint
    end

    λ_int = IntervalArithmetic.interval(λ)
    Q_int = IntervalMatrices.IntervalMatrix(IntervalArithmetic.interval.(Q))
    @info "Checking in $(eltype(Q_int)) arithmetic with" λ_int

    check, sos_int = @time if should_we_augment
        @info("Projecting columns of Q to the augmentation ideal...")
        Q_int = augment_columns!(Q_int)
        @info "Checking that sum of every column contains 0.0..."
        check_augmented = all(0 ∈ sum(c) for c in eachcol(Q_int))
        check_augmented || @error(
            "Augmentation failed! The following numbers are not certified!"
        )
        sos_int = compute_sos(parent(elt), Q_int; augmented = augmented)
        check_augmented, sos_int
    else
        true, compute_sos(parent(elt), Q_int; augmented = augmented)
    end

    λ_certified =
        sufficient_λ(elt, orderunit, λ_int, sos_int; halfradius = halfradius)

    return check && IntervalArithmetic.inf(λ_certified) > 0.0, λ_certified
end
