#=
    Example 1 
    - "Stochastic Lipschitz dynamic programming," Ahmed et al. (2022)
    - state: 1D continuous
    - local: continuous + binary
    - The feasible region is discrete due to the equality constraint:
        y.out = y.in + h - 1
        or
        y.out = y.in + h + 1.
=#

using JuMP
using Gurobi
using ScaledCutNBD
import Dates: now, format

const Ω = [
    [0.],
    [0.9393 0.9372 0.8748 0.9121 0.9472 -0.6611 -0.9336 -0.9929 -0.9091 -0.8857],
    [0.9634 0.8446 0.5839 0.5108 0.6148 -0.731 -0.8563 -0.8902 -0.6424 -0.526],
    [0.6066 0.56 0.5431 0.8875 0.9185 -0.9492 -0.9462 -0.5121 -0.8931 -0.692],
    [0.7446 0.7032 0.7029 0.9353 0.8546 -0.77 -0.9358 -0.6878 -0.9622 -0.8485],
]
const y_LB = -3.
const y_UB = 3.
const num_branch = [1, 10, 10, 10, 10]
const discount_factor = 0.9
const M = 10.

# Options
const ITER_LIMIT = 100
const TIME_LIMIT = 300.
const LOG_FILE = nothing
# const LOG_FILE = "log_ex_01_1D_$(format(now(), "yyyy-mm-dd_HHMMSS")).txt"
const NUM_SAMPLE_PATHS = 1
const NUM_SIMULATIONS = 0
const CGSP_TOL = 1e-2

function build_model(r::Int)
    env = Gurobi.Env()
    optimizer = () -> Gurobi.Optimizer(env)

    # Build the tree
    tree = Tree(num_branch, r, optimizer, 0.) do sub::JuMP.Model, t::Int
        @variable(sub, y_LB <= y <= y_UB, State, initial_value = 2.0)

        @variable(sub, y_plus >= 0)
        @variable(sub, y_minus >= 0)
        @variable(sub, z, Bin) # (2z - 1) ∈ {-1, 1}
        @variable(sub, h)

        @constraint(sub, y.out == y_plus - y_minus)
        @constraint(sub, y_plus + y_minus <= M)
        @constraint(sub, y.out - y.in - (2z - 1) == h)

        obj_sense = MIN_SENSE
        stage_obj_function = @expression(sub, (discount_factor)^(t-1) * (y_plus + y_minus))

        function parameterize(t::Int, n::Int)
            fix(h, Ω[t][n])
        end

        return obj_sense, stage_obj_function, parameterize
    end

    return NBD(
        tree; 
        iteration_limit = ITER_LIMIT, 
        time_limit = TIME_LIMIT,
        num_sample_paths=NUM_SAMPLE_PATHS, 
        num_simulations=NUM_SIMULATIONS,
        log_file=LOG_FILE,
        cgsp_tol=CGSP_TOL,
    )
end

function main()
    if length(ARGS) == 0
        println("Usage: julia ex_01_1D.jl <r>")
        return
    elseif length(ARGS) > 1
        println("Too many arguments. Usage: julia ex_01_1D.jl <r>")
        return
    end
    r = parse(Int, ARGS[1])
    if r > 5 || r < 1
        error("r must be between 1 and 5")
    end

    nbd = build_model(r)
    ScaledCutNBD.run(nbd)

    println("\nTimer Output:")
    println(nbd.runstats.timer_output)
end

main()