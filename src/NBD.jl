function NBD(
    tree::Tree
    ; 
    iteration_limit::Union{Nothing, Int}=nothing, 
    time_limit::Union{Nothing, Float64}=nothing, 
    num_sample_paths::Int=1,
    num_simulations::Int=0,
    log_file::Union{Nothing, String}=nothing,
    alpha::Float64=0.05, # for confidence interval
    # CG parameters
    cg_bound::Float64 = 1e4,
    cg_time_limit::Float64 = 30.0,
    cgsp_tol::Float64 = 1e-4,
    fixed_iter_tol::Float64 = 1e-4,
    regularization_param::Float64 = 1e-6,
    cg_acc_until::Int = 100,
    cg_acc_level::Int = 1,
    convergence_check_iterations::Int = 0,
    convergence_check_tol::Float64 = 1e-4,
)::NBD
    options = Options(
        log_file, 
        iteration_limit, 
        time_limit, 
        num_sample_paths, 
        num_simulations,
        alpha,
        cg_bound,
        cg_time_limit,
        cgsp_tol,
        fixed_iter_tol,
        regularization_param,
        cg_acc_until,
        cg_acc_level,
        1e-6, # obj_diff_tol when adding cuts
        convergence_check_iterations, # convergence_check_iterations (default=0: disabled)
        convergence_check_tol, # convergence_check_tol
        10000, # cg_inner_iteration_limit
    )

    # Build CutGenerators
    build_cut_generators(tree.model_info.cut_generator_type, tree, options)

    nbd = NBD(
        tree, 
        RunStats(), 
        options,
    )

    # Attach NBD instance to each node for easy access
    for node in values(tree.nodes)
        node.ext[:nbd] = nbd
    end
    
    return nbd
end

function run(nbd::NBD)
    reset_timer!(timer_output)
    
    io = nbd.options.log_file !== nothing ? open(nbd.options.log_file, "w") : stdout
    try
        print_header(io, nbd)
        # Main loop
        run_start_time = time()
        while true
            iteration_result = iteration(nbd)
            push!(nbd.runstats.iterations, iteration_result)
            print_result(io, iteration_result)
            if check_termination(io, nbd)
                break
            end
        end
        nbd.runstats.ext[:total_run_time] = time() - run_start_time
        print_tailer(io, nbd)
    catch e
        if e isa InterruptException
            @info "NBD run interrupted."
            # rethrow(e)
        else
            rethrow(e)
        end
    finally
        if nbd.options.log_file !== nothing
            close(io)
        end
    end
    return
end

function iteration(nbd::NBD)
    sample_paths = [get_sample_path(nbd.tree.model_info.num_branch) for _ in 1:nbd.options.num_sample_paths]
    forward_results = Vector{Dict{Symbol, Any}}(undef, nbd.options.num_sample_paths)

    for (i, sample_path) in enumerate(sample_paths)
        @timeit timer_output "forward_pass" begin
            forward_results[i] = forward_pass(nbd, sample_path)
        end
    end

    # Compute the statistical primal bound estimate
    # μ̂ + t_{α/2, L-1} * (σ̂ / sqrt(L))
    # If num_sample_paths = 1, then return the single sample path objective value.
    primal_bound_estimate = compute_primal_bound_estimate(nbd, forward_results)

    for (i, sample_path) in enumerate(sample_paths)
        @timeit timer_output "backward_pass" begin
            backward_pass(nbd, sample_path, forward_results[i])
        end
    end

    # Update the dual bound
    dual_bound = compute_dual_bound(nbd.tree.root_node)

    return Dict(
        :iteration => length(nbd.runstats.iterations) + 1,
        :primal_bound_estimate => primal_bound_estimate,
        :dual_bound => dual_bound,
        :elapsed_time => get_time_sec(timer_output),
    )
end

function check_termination(io::IO, nbd::NBD)
    # iteration_limit
    if nbd.options.iteration_limit !== nothing && length(nbd.runstats.iterations) >= nbd.options.iteration_limit
        println(io, "Iteration limit reached.")
        return true
    end
    # time_limit
    if nbd.options.time_limit !== nothing 
        if get_time_sec(timer_output) >= nbd.options.time_limit
            println(io, "Time limit reached.")
            return true
        end
    end
    # convergence check
    if nbd.options.convergence_check_iterations > 0 && length(nbd.runstats.iterations) > nbd.options.convergence_check_iterations
        dual_bound_diff = abs(nbd.runstats.iterations[end][:dual_bound] - nbd.runstats.iterations[end - nbd.options.convergence_check_iterations][:dual_bound])
        dual_bound_gap = dual_bound_diff / (abs(nbd.runstats.iterations[end][:dual_bound]) + 1e-10)
        if dual_bound_gap <= nbd.options.convergence_check_tol
            println(io, "Bound converged.")
            return true
        end
    end
    return false
end

function forward_pass(nbd::NBD, sample_path::Tuple{Vararg{Int}})::Dict{Symbol, Any}
    # Sampled states along the path. Dict(State_name => Value)
    forward_states = Dict{Symbol,Float64}()
    # Initialize the initial states at the root node
    for state in values(nbd.tree.root_node.subproblem.state_vars)
        name = Symbol(state.original_name, "_0")
        forward_states[name] = JuMP.fix_value(state.in)
    end
    # Last sampled states to be used as incoming states for the next node. Dict(Original_state_name => Value)
    last_states = Dict{Symbol,Float64}()
    cumulative_stage_obj = 0.0
    # Outer approximation values at stages along the path (t = 1, ..., T, ϕ(T) = 0.)
    theta_values = zeros(Float64, length(sample_path))

    node = nothing
    for (t, branch_idx) in enumerate(sample_path)
        if t == 1
            node = nbd.tree.root_node
        else
            node = node.children[branch_idx]
        end
        
        if t > 1
            # fix state variables to last_states
            for (name, state) in node.subproblem.state_vars
                if haskey(last_states, state.original_name)
                    fix(state.in, last_states[state.original_name])
                else
                    error("State variable $(state.original_name) not found in last_states during forward_pass. \nlast_states: $last_states")
                end
            end
            # fix extended state variables to forward_states
            for (name, var) in node.subproblem.extend_vars
                if haskey(forward_states, name)
                    fix(var, forward_states[name])
                else
                    error("Extended-space variable $name not found in subproblem during forward_pass.")
                end
            end
        end
        # parameterize the subproblem
        node.subproblem.parameterize(branch_idx)
        # Solve the stage subproblem
        TimerOutputs.@timeit timer_output "stage_subproblem" begin
            solve!(node.subproblem.model)    
        end # timer
        # Get the outgoing state values
        for (name, state) in node.subproblem.state_vars
            out_value = JuMP.value(state.out)
            # prevent numerical issue
            if is_integer(state.out) || is_binary(state.out)
                # throw error if not close to an integer
                rounded_out_value = round(out_value)
                if isapprox(out_value, rounded_out_value; atol=1e-3) == false
                    error("State variable $(state.original_name) has non-integer value $out_value at node $(node.path) during forward_pass. It might be an error in the model formulation or solver accuracy.")
                end
                out_value = rounded_out_value
            end
            last_states[state.original_name] = out_value
            if haskey(forward_states, name)
                error("Variable $name already exists in forward_states during forward_pass.")
            end
            forward_states[name] = out_value
        end
        # Get the outer approximation value (theta) if not a leaf node
        if node.depth < length(sample_path)
            theta_values[t] = JuMP.value(node.subproblem.theta)
        end
        
        # Get the subproblem stage_objective_value
        cumulative_stage_obj += value(node.subproblem.stage_objective)
    end

    return Dict(
        :cumulative_stage_obj => cumulative_stage_obj,
        :forward_states => forward_states,
        :theta_values => theta_values,
    )
end

function backward_pass(nbd::NBD, sample_path::Tuple{Vararg{Int}}, forward_result::Dict{Symbol, Any})
    # Get the leaf node
    node = nbd.tree.nodes[Tuple(sample_path)]
    t = length(sample_path)
    while node.parent !== nothing
        t -= 1
        node = node.parent
        # Generate cut using the method defined in CutGenerator.
        TimerOutputs.@timeit timer_output "generating_cuts" begin
            cut_info = generate_cut(nbd.tree.model_info.cut_generator_type, forward_result[:forward_states], forward_result[:theta_values][t], node, nbd.tree.model_info, nbd.options)
        end
        # If the cut is not helping (obj_diff_tol), skip adding it.
        if cut_info[:evaluated_obj] - forward_result[:theta_values][t] <= nbd.options.obj_diff_tol
            continue
        end
        # Add the cut to the node's convex approximation
        TimerOutputs.@timeit timer_output "adding_cuts" begin
            # Normal cut addition
            for model_hook in node.convex_approx.model_hooks
                model_hook.add_cut_func(cut_info[:slope], cut_info[:intercept])
            end
            # We track the cut data (α, β, τ) for analysis and cut_write.
            # We may remove this later if not needed.
            push!(node.convex_approx.cuts, ScaledCut(cut_info[:α], cut_info[:β], cut_info[:τ])) 
        end
    end
end

function compute_dual_bound(root_node::Node)::Float64
    root_node_problem = root_node.subproblem.model
    solve!(root_node_problem)
    return JuMP.objective_value(root_node_problem)
end

function compute_primal_bound_estimate(nbd::NBD, forward_results::Vector{Dict{Symbol, Any}})::Float64
    if nbd.options.num_simulations > 0
        # Conduct simulations
        simulation_results = Vector{Dict{Symbol, Any}}(undef, nbd.options.num_simulations)
        for sim in 1:nbd.options.num_simulations
            sim_sample_path = get_sample_path(nbd.tree.model_info.num_branch)
            @timeit timer_output "simulation_forward_pass" begin
                simulation_results[sim] = forward_pass(nbd, sim_sample_path)
            end
        end
        t_critical = compute_t_critical(nbd.options.num_simulations, nbd.options.alpha)
        return compute_CI_UB(simulation_results, t_critical) 
    else
        t_critical = nbd.options.num_sample_paths > 1 ? compute_t_critical(nbd.options.num_sample_paths, nbd.options.alpha) : 0.
        return compute_CI_UB(forward_results, t_critical)
    end
end

function compute_CI_UB(forward_results::Vector{Dict{Symbol, Any}}, t_critical::Float64)::Float64
    if t_critical == 0.
        # No CI computation, return the single sample path objective value.
        return forward_results[1][:cumulative_stage_obj]
    end
    L = length(forward_results)
    stage_obj_values = [forward_results[i][:cumulative_stage_obj] for i in 1:L]
    μ̂ = mean(stage_obj_values)
    σ̂ = std(stage_obj_values)
    CI_upper_bound = μ̂ + t_critical * (σ̂ / sqrt(L))
    return CI_upper_bound
end

function objective_bound(nbd::NBD)
    return nbd.runstats.iterations[end][:dual_bound]
end