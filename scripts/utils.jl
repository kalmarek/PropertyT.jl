using Dates
using Serialization
using Logging

import JuMP

function get_solution(model)
    λ = JuMP.value(model[:λ])
    Q = real.(sqrt(JuMP.value.(model[:P])))
    solution = Dict(:λ => λ, :Q => Q)
    return solution
end

function get_solution(model, wd, varP, eps = 1e-10)
    λ = JuMP.value(model[:λ])

    @info "reconstructing the solution"
    Q = @time let wd = wd, Ps = [JuMP.value.(P) for P in varP], eps = eps
        PropertyT.__droptol!.(Ps, 100eps)
        Qs = real.(sqrt.(Ps))
        PropertyT.__droptol!.(Qs, eps)
        PropertyT.reconstruct(Qs, wd)
    end

    solution = Dict(:λ => λ, :Q => Q)

    return solution
end

function solve_in_loop(model::JuMP.Model, args...; logdir, optimizer, data)
    @info "logging to $logdir"
    status = JuMP.UNKNOWN_RESULT_STATUS
    warm = try
        solution = deserialize(joinpath(logdir, "solution.sjl"))
        warm = solution[:warm]
        @info "trying to warm-start model with λ=$(solution[:λ])..."
        warm
    catch
        nothing
    end
    old_lambda = 0.0
    while status != JuMP.OPTIMAL
        date = now()
        log_file = joinpath(logdir, "solver_$date.log")
        @info "Current logfile is $log_file."
        isdir(dirname(log_file)) || mkpath(dirname(log_file))

        λ, flag, certified_λ = let
            # logstream = current_logger().logger.stream
            # v = @ccall setvbuf(logstream.handle::Ptr{Cvoid}, C_NULL::Ptr{Cvoid}, 1::Cint, 0::Cint)::Cint
            # @warn v
            status, warm =
                @time PropertyT.solve(log_file, model, optimizer, warm)

            solution = get_solution(model, args...)
            serialize(joinpath(logdir, "solution_$date.sjl"), solution)
            serialize(joinpath(logdir, "solution.sjl"), solution)

            solution[:warm] = warm

            flag, λ_cert = open(log_file; append = true) do io
                with_logger(SimpleLogger(io)) do
                    return PropertyT.certify_solution(
                        data.elt,
                        data.unit,
                        solution[:λ],
                        solution[:Q];
                        halfradius = data.halfradius,
                    )
                end
            end

            solution[:λ], flag, λ_cert
        end

        if flag == true && certified_λ ≥ 0
            @info "Certification done with λ = $certified_λ" certified_λ rel_change status
            return certified_λ
        else
            rel_change =
                abs(certified_λ - old_lambda) /
                (abs(certified_λ) + abs(old_lambda))
            @info "Certification failed with λ = $λ" certified_λ rel_change status
        end

        old_lambda = certified_λ

        if rel_change < 1e-9
            @info "No progress detected, breaking" certified_λ rel_change status
            break
        end
    end

    return status == JuMP.OPTIMAL ? old_lambda : NaN
end
