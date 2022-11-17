using ArgParse

args_settings = ArgParseSettings()
@add_arg_table! args_settings begin
    "-N"
    help = "the degree/genus/etc. parameter for a group"
    arg_type = Int
    default = 3
    "--halfradius", "-R"
    help = "the halfradius on which perform the sum of squares decomposition"
    arg_type = Int
    default = 2
    "--upper_bound", "-u"
    help = "set upper bound for the optimization problem to speed-up the convergence"
    arg_type = Float64
    default = Inf
end

parsed_args = parse_args(ARGS, args_settings)
