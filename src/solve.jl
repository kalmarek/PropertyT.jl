## Low-level solve

setwarmstart!(model::JuMP.Model, ::Nothing) = model

function setwarmstart!(model::JuMP.Model, warmstart)
    constraint_map = Dict(
        ct => JuMP.all_constraints(model, ct...) for
        ct in JuMP.list_of_constraint_types(model)
    )
    try
        JuMP.set_start_value.(JuMP.all_variables(model), warmstart.primal)

        for (ct, idx) in pairs(constraint_map)
            JuMP.set_start_value.(idx, warmstart.slack[ct])
            JuMP.set_dual_start_value.(idx, warmstart.dual[ct])
        end
    catch e
        @warn "Warmstarting was not succesfull!" e
    end
    return model
end

function getwarmstart(model::JuMP.Model)
    constraint_map = Dict(
        ct => JuMP.all_constraints(model, ct...) for
        ct in JuMP.list_of_constraint_types(model)
    )

    primal = value.(JuMP.all_variables(model))

    slack = Dict(k => value.(v) for (k, v) in constraint_map)
    duals = Dict(k => JuMP.dual.(v) for (k, v) in constraint_map)

    return (primal = primal, dual = duals, slack = slack)
end

function solve(m::JuMP.Model, optimizer, warmstart = nothing)
    JuMP.set_optimizer(m, optimizer)
    MOIU.attach_optimizer(m)

    m = setwarmstart!(m, warmstart)

    JuMP.optimize!(m)

    status = JuMP.termination_status(m)

    return status, getwarmstart(m)
end

function solve(solverlog::String, m::JuMP.Model, optimizer, warmstart = nothing)
    isdir(dirname(solverlog)) || mkpath(dirname(solverlog))

    Base.flush(Base.stdout)
    Base.Libc.flush_cstdio()
    status, warmstart = open(solverlog, "a+") do logfile
        redirect_stdout(logfile) do
            status, warmstart = solve(m, optimizer, warmstart)
            Base.Libc.flush_cstdio()
            return status, warmstart
        end
    end

    return status, warmstart
end
