function check_positivity(
    elt,
    unit;
    upper_bound = Inf,
    halfradius = 2,
    optimizer,
)
    @time sos_problem =
        PropertyT.sos_problem_primal(elt, unit; upper_bound = upper_bound)

    status, _ = PropertyT.solve(sos_problem, optimizer)
    P = JuMP.value.(sos_problem[:P])
    Q = real.(sqrt(P))
    certified, λ_cert = PropertyT.certify_solution(
        elt,
        unit,
        JuMP.objective_value(sos_problem),
        Q;
        halfradius = halfradius,
    )
    return status, certified, λ_cert
end

function check_positivity(
    elt,
    unit,
    wd;
    upper_bound = Inf,
    halfradius = 2,
    optimizer,
)
    @assert SA.aug(elt) == SA.aug(unit) == 0
    @time sos_problem, Ps =
        PropertyT.sos_problem_primal(elt, unit, wd; upper_bound = upper_bound)

    @time status, _ = PropertyT.solve(sos_problem, optimizer)

    Q = let Ps = Ps
        Qs = [real.(sqrt(JuMP.value.(P))) for P in Ps]
        PropertyT.reconstruct(Qs, wd)
    end

    λ = JuMP.value(sos_problem[:λ])

    certified, λ_cert =
        PropertyT.certify_solution(elt, unit, λ, Q; halfradius = halfradius)
    return status, certified, λ_cert
end
