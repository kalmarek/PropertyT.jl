## Optimizers

import JuMP
import SCS

function scs_optimizer(;
    accel = 10,
    alpha = 1.5,
    eps = 1e-9,
    max_iters = 100_000,
    verbose = true,
    linear_solver = SCS.DirectSolver,
)
    return JuMP.optimizer_with_attributes(
        SCS.Optimizer,
        "acceleration_lookback" => accel,
        "acceleration_interval" => 10,
        "alpha" => alpha,
        "eps_abs" => eps,
        "eps_rel" => eps,
        "linear_solver" => linear_solver,
        "max_iters" => max_iters,
        "warm_start" => true,
        "verbose" => verbose,
    )
end

import COSMO

function cosmo_optimizer(;
    accel = 15,
    alpha = 1.6,
    eps = 1e-9,
    max_iters = 100_000,
    verbose = true,
    verbose_timing = verbose,
    decompose = false,
)
    return JuMP.optimizer_with_attributes(
        COSMO.Optimizer,
        "accelerator" =>
            COSMO.with_options(COSMO.AndersonAccelerator; mem = max(accel, 2)),
        "alpha" => alpha,
        "decompose" => decompose,
        "eps_abs" => eps,
        "eps_rel" => eps,
        "eps_prim_inf" => eps,
        "eps_dual_inf" => eps,
        "max_iter" => max_iters,
        "verbose" => verbose,
        "verbose_timing" => verbose_timing,
        "check_termination" => 250,
    )
end
