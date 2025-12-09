"""
    Get the zeroed-out path associated with `node_path`.
    - `node_path`: The original node path as a tuple of integers.
    - `T`: Total number of stages.
    - `t`: Current stage of the node.
    - `r`: The depth for the extended space.

    * The first `t - r + 1` elements are zeroed out to represent the shared outer approximation / subproblem.
    * e.g.) 
        t=3, r=1. (1, 2, 2) -> (0, 0, 0), 
            # current_t doesn't affect the cut for φₙ. So, we can share the same subproblem for all nodes at t=3 (with `parameterized` data).
        t=3, r=2. (1, 2, 2) -> (0, 0, 2),
            # Now the cut for φₙ depends on ξ₃.
"""
function get_zeroed_out_path(node_path::Tuple{Vararg{Int}}, r::Int)
    # Zero out the first (T - r + 1) elements
    tmp_path = collect(node_path)
    tmp_path[1:(end - r + 1)] .= 0
    return Tuple(tmp_path)
end


"""
    Sample a random path from the scenario tree defined by `num_branch`.
    - `num_branch`: A vector indicating the number of branches (realizations) at each stage.
"""
function get_sample_path(num_branch::Vector{Int})::Tuple{Vararg{Int}}
    path = []
    for t in 1:length(num_branch)
        push!(path, rand(1:num_branch[t]))
    end
    return Tuple(path)
 end

 function project_to_bounds(var::JuMP.VariableRef)
    # project down to the bounds if necessary
    value = value(var)
    if JuMP.has_upper_bound(var)
        ub = JuMP.upper_bound(var)
        value = ub < value ? ub : value
    end
    if JuMP.has_lower_bound(var)
        lb = JuMP.lower_bound(var)
        value = lb > value ? lb : value
    end
    return value
end

function get_time_sec(to::TimerOutput, sec_name::String)
    return TimerOutputs.time(to[sec_name]) / 1e9
end

function get_time_sec(to::TimerOutput)
    return TimerOutputs.tottime(to) / 1e9
end

function solve!(model::JuMP.Model)
    optimize!(model)

    is_solved_and_feasible
    if primal_status(model) ∉ (FEASIBLE_POINT, NEARLY_FEASIBLE_POINT)
        if primal_status(model) == UNKNOWN_RESULT_STATUS
            try
                # try to access variable value. If okay, then the model is feasible.
                value(all_variables(model)[1])
            catch
                error(
                    "\nInfeasible model during solve!:\n" * 
                    "The primal status: $(JuMP.primal_status(model))\n" *
                    "The termination status: $(JuMP.termination_status(model))\n" *
                    "$(model)\n" *
                    ""
                )
            end
        else
            error(
                "\nInfeasible model during solve!:\n" * 
                "The primal status: $(JuMP.primal_status(model))\n" *
                "The termination status: $(JuMP.termination_status(model))\n" *
                "$(model)\n" *
                ""
            )
        end
    end
end

function compute_t_critical(num_sample_paths::Int, alpha::Float64)
    df = num_sample_paths - 1
    return quantile(TDist(df), 1.0 - alpha / 2)
end

const PRINT_PADS = [6, 15, 15, 10]

# maintain the values have the same format same length
function num_format(value::Float64)
    return @sprintf("%.5f", value)
end

function time_format(value::Float64)
    return @sprintf("%.2f", value)
end

function print_header(io::IO, nbd::NBD)
    println(io, "================================================")
    println(io, "Scaled Cut based Nested Benders Decomposition\n" *
                "ScaledCutNBD.jl version $VERSION")
    println(io, "------------------------------------------------")
    # print information about the problem and tree
    println(io, "System Information")
    println(io, "  CPU                    : $(Sys.CPU_NAME) ($(Sys.cpu_info()[1].speed / 1000) GHz)")
    println(io, "  Threads (Julia)        : ", Threads.nthreads())
    println(io, "  RAM (Physical)         : $(round(Sys.total_memory() / 1024^3, digits=2)) GB")
    println(io, "Problem/Tree Information")
    println(io, "  r                      : ", nbd.tree.model_info.r)
    println(io, "  #stages                : ", nbd.tree.model_info.num_stage)
    println(io, "  #branches              : ", nbd.tree.model_info.num_branch)
    println(io, "  #nodes                 : ", length(nbd.tree.nodes))
    println(io, "  #subproblems           : ", length(nbd.tree.subproblems))
    println(io, "  #convex approximations : ", length(nbd.tree.convex_approxs))
    println(io, "  #cut generators        : ", length(nbd.tree.cut_generators))
    println(io, "  cut generator type     : ", nbd.tree.model_info.cut_generator_type)
    println(io, "  cut read from file     : ", get(nbd.runstats.ext, :cut_file_name, "None"))
    println(io, "Options")
    println(io, "  iteration limit        : ", nbd.options.iteration_limit)
    println(io, "  time limit             : ", nbd.options.time_limit)
    println(io, "  #sample paths          : ", nbd.options.num_sample_paths)
    println(io, "  CG bound               : ", nbd.options.cg_bound)
    println(io, "  CG time limit          : ", nbd.options.cg_time_limit)
    println(io, "  CGSP tolerance         : ", nbd.options.cgsp_tol)
    println(io, "  fix-iter alg tol       : ", nbd.options.fixed_iter_tol)
    println(io, "  reg multiplier         : ", nbd.options.regularization_param)
    println(io, "  #simulations           : ", nbd.options.num_simulations)
    println(io, "  CI alpha               : ", nbd.options.alpha)
    println(io, "Date: $(format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io, "")
    println(io, "------------------------------------------------")
    println(io, "Test Results")
    println(io, "------------------------------------------------")

    header = (
        "Iter",
        "Obj Estimate",
        "Dual Bound",
        "Time (s)"
    )
    for (header_str, pad) in zip(header, PRINT_PADS)
        print(io, lpad(header_str, pad))
    end
    println(io, "")
    flush(io)
end

function print_result(io::IO, iteration_result::Dict)
    row = (
        iteration_result[:iteration],
        num_format(iteration_result[:primal_bound_estimate]),
        num_format(iteration_result[:dual_bound]),
        time_format(iteration_result[:elapsed_time])
    )
    println(io, lpad(row[1], PRINT_PADS[1]), lpad(row[2], PRINT_PADS[2]), lpad(row[3], PRINT_PADS[3]), lpad(row[4], PRINT_PADS[4]))
    flush(io)
end

function print_tailer(io::IO, nbd::NBD)
    println(io, "------------------------------------------------")
    println(io, "Total solve runtime      : $(time_format(get_time_sec(nbd.runstats.timer_output))) sec")
    println(io, "Total wall-clock runtime : $(time_format(nbd.runstats.ext[:total_run_time])) sec")
    println(io, "Best bound               : ", num_format(nbd.runstats.iterations[end][:dual_bound]))
    println(io, "================================================")
end

function Base.show(io::IO, d::Dict{K, V}) where {K, V}
    print(io, 
        "Dict{$K, $V} with $(length(d)) entries: (\n" *
        join([ "  $k => $v" for (k, v) in d ], ",\n") *
        ")"
    )
end