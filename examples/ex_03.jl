#=
    ex_03.jl
    Multistage stochastic lot-sizing problem with backlogging
=#

using Random
using Distributions
using JuMP, Gurobi
using Dates: now, format
using ScaledCutNBD
include("ex_03_parser.jl")
using .Parser

using SDDP


"""
    Generate instance data
"""
function generate_data(
    num_branch::Vector{Int},
    num_item::Int,
    int_quantity_flag::Int,
    ;
    demand_range::Vector{Tuple{Float64, Float64}}=[(1.0, 10.0) for _ in 1:num_item],
    production_cost_range::Vector{Tuple{Float64, Float64}}=[(0.2, 0.4) for _ in 1:num_item],
    holding_cost_range::Vector{Tuple{Float64, Float64}}=[(0.05, 0.1) for _ in 1:num_item],
    fixed_cost::Float64=2.0,
    backlogging_cost_range::Vector{Tuple{Float64, Float64}}=[(0.05, 0.1) for _ in 1:num_item],
    seed::Int=1,
    round_digits::Int=2,
)::NamedTuple
    round_val(x) = round.(x; digits=round_digits)

    T = length(num_branch)
    Random.seed!(seed)  
    # random demand
    demand_dist = int_quantity_flag == 1 ? DiscreteUniform : Uniform
    demand = [
        [rand(demand_dist(demand_range[i][1], demand_range[i][2]), num_branch[t]) |> round_val for i in 1:num_item] 
    for t in 1:T]
    demand_mean = [(demand_range[i][1] + demand_range[i][2]) / 2.0 for i in 1:num_item]

    # deterministic costs and capacities
    production_cost = [[rand(Uniform(production_cost_range[i][1], production_cost_range[i][2])) |> round_val for i in 1:num_item] for _ in 1:T]

    production_capacity = sum(demand_mean) * 2.0

    fixed_cost = [fixed_cost for _ in 1:num_item]
    holding_cost = [[rand(Uniform(holding_cost_range[i][1], holding_cost_range[i][2])) |> round_val for i in 1:num_item] for _ in 1:T]
    backlogging_cost = [[rand(Uniform(backlogging_cost_range[i][1], backlogging_cost_range[i][2])) |> round_val for i in 1:num_item] for _ in 1:T]
    # penalize the last-stage's backlogging cost
    for i in 1:num_item
        backlogging_cost[T][i] = 1000.0
    end
    # inventory capacity
    inventory_capacity = [[t * production_capacity for _ in 1:num_item] for t in 1:T]
    return (
        demand=demand,
        production_cost=production_cost,
        fixed_cost=fixed_cost,
        production_capacity=production_capacity,
        holding_cost=holding_cost,
        backlogging_cost=backlogging_cost,
        num_branch=num_branch,
        num_item=num_item,
        inventory_capacity=inventory_capacity,
    )
end

function build_scnbd(
    data::NamedTuple,
    r::Int,
    theta_bound::Float64,
    cg_type::Type{<:ScaledCutNBD.AbstractCutGenerator},
    binarized_state_flag::Int,
    decimal_factor::Int,
    int_quantity_flag::Int,
)
    # determine binarization parameters
    M = maximum([maximum(cap) for cap in data.inventory_capacity])
    discrete_factor = int_quantity_flag == 1 ? 1 : 10^decimal_factor
    n::Int = floor(log2(M * discrete_factor)) + 1 # number of binary variables required to represent the state variable
    if binarized_state_flag == 1
        @info "Binarization parameters" M n
    end

    num_branch = data.num_branch
    num_item = data.num_item
    # build tree
    env = Gurobi.Env()
    optimizer = () -> Gurobi.Optimizer(env)
    tree = Tree(
        num_branch, 
        r, 
        optimizer, 
        theta_bound,
        cut_generator_type=cg_type,
    ) do sub, t
        s_out = Vector{VariableRef}(undef, num_item)
        s_in = Vector{VariableRef}(undef, num_item)
        b_out = Vector{VariableRef}(undef, num_item)
        b_in = Vector{VariableRef}(undef, num_item)

        if binarized_state_flag == 1
            @variable(sub, 0 <= s_bin[i in 1:num_item, k in 1:n], State, Bin, initial_value=0)
            @variable(sub, 0 <= b_bin[i in 1:num_item, k in 1:n], State, Bin, initial_value=0)

            for i in 1:num_item
                s_out[i] = @variable(sub, base_name="s_out_$(i)")
                s_in[i] = @variable(sub, base_name="s_in_$(i)")
                b_out[i] = @variable(sub, base_name="b_out_$(i)")
                b_in[i] = @variable(sub, base_name="b_in_$(i)")
            end
            @constraint(sub, [i in 1:num_item], s_out[i] == sum(s_bin[i, k].out * 2.0^(k-1) / discrete_factor for k in 1:n))
            @constraint(sub, [i in 1:num_item], s_in[i] == sum(s_bin[i, k].in * 2.0^(k-1) / discrete_factor for k in 1:n))
            @constraint(sub, [i in 1:num_item], b_out[i] == sum(b_bin[i, k].out * 2.0^(k-1) / discrete_factor for k in 1:n))
            @constraint(sub, [i in 1:num_item], b_in[i] == sum(b_bin[i, k].in * 2.0^(k-1) / discrete_factor for k in 1:n))
        else
            if int_quantity_flag == 1
                @variable(sub, 0 <= s[i in 1:num_item] <= data.inventory_capacity[t][i], State, Int, initial_value=0.)
                @variable(sub, 0 <= b[i in 1:num_item] <= data.inventory_capacity[t][i], State, Int, initial_value=0.)
            else
                @variable(sub, 0 <= s[i in 1:num_item] <= data.inventory_capacity[t][i], State, initial_value=0.)
                @variable(sub, 0 <= b[i in 1:num_item] <= data.inventory_capacity[t][i], State, initial_value=0.)
            end
            s_in = [s[i].in for i in 1:num_item]
            s_out = [s[i].out for i in 1:num_item]
            b_in = [b[i].in for i in 1:num_item]
            b_out = [b[i].out for i in 1:num_item]
        end

        if int_quantity_flag == 1
            @variable(sub, 0 <= x[1:num_item] <= data.production_capacity, Int)
        else
            @variable(sub, 0 <= x[1:num_item] <= data.production_capacity)
        end
        @variable(sub, y[1:num_item], Bin)
        @variable(sub, ω[1:num_item])

        @constraint(sub, [i in 1:num_item], x[i] <= data.production_capacity * y[i])
        @constraint(sub, sum(x[i] for i in 1:num_item) <= data.production_capacity)
        @constraint(sub, [i in 1:num_item], b_out[i] - s_out[i] + x[i] == ω[i] + b_in[i] - s_in[i])
        
        obj_sense = MIN_SENSE
        stage_obj_function = @expression(sub, 
            sum(
                data.production_cost[t][i] * x[i] +
                data.fixed_cost[i] * y[i] +
                data.holding_cost[t][i] * s_out[i] +
                data.backlogging_cost[t][i] * b_out[i]
                for i in 1:num_item
            )
        )

        function parameterize(t::Int, n::Int)
            for i in 1:num_item
                fix(ω[i], data.demand[t][i][n])
            end
        end

        return obj_sense, stage_obj_function, parameterize
    end
    return tree
end

function run_scnbd(
    data::NamedTuple,
    scnbd_params::NamedTuple,
    print_all_node_models::Bool,
)
    if scnbd_params.cut_io_mode ∉ (:n, :r, :w, :rw)
        error("Invalid cut_io_mode: $(scnbd_params.cut_io_mode). Must be :n, :r, :w, or :rw.")
    end

    tree = build_scnbd(data, scnbd_params.r, scnbd_params.theta_bound, scnbd_params.cg_type, scnbd_params.binarized_state_flag, scnbd_params.decimal_factor, scnbd_params.int_quantity_flag)
    nbd = NBD(
        tree; 
        iteration_limit=scnbd_params.iteration_limit, 
        time_limit=scnbd_params.time_limit, 
        log_file=scnbd_params.log_file, 
        num_sample_paths=1, 
        num_simulations=0, 
        cg_bound=scnbd_params.cg_bound, 
        cgsp_tol=scnbd_params.cgsp_tol,
        regularization_param=scnbd_params.regularization_param,
        fixed_iter_tol=scnbd_params.fixed_iter_tol,
        cg_acc_until=scnbd_params.cg_acc_until,
        cg_acc_level=scnbd_params.cg_acc_level,
        convergence_check_iterations=0,
    )

    # Read cuts
    if scnbd_params.cut_io_mode in (:r, :rw) && scnbd_params.cut_read_filename !== nothing
        ScaledCutNBD.read_cuts(nbd, scnbd_params.cut_read_filename)
    end

    ScaledCutNBD.run(nbd)
    @info "\nSCNBD Best Bnd: $(ScaledCutNBD.objective_bound(nbd))"

    # Write cuts
    if scnbd_params.cut_io_mode in (:w, :rw) && scnbd_params.cut_write_filename !== nothing
        ScaledCutNBD.write_cuts(nbd, scnbd_params.cut_write_filename)
    end

    # print timeroutput
    if scnbd_params.print_timer_output
        println("\n===== Timer Output =====")
        println(nbd.runstats.timer_output)
    end

    if print_all_node_models
        for (path, node) in tree.nodes
            @info "Node path: $path"
            println(node.subproblem.model)
            println()
        end
    end
end

function build_def(
    data::NamedTuple,
    time_limit::Float64,
    int_quantity_flag::Int,
)
    function all_sample_paths(num_branch::Vector{Int})::Vector{Vector{Int}}
        T = length(num_branch)
        total_paths = prod(num_branch)
        paths = Vector{Vector{Int}}(undef, total_paths)
        for p in 1:total_paths
            path = Vector{Int}(undef, T)
            idx = p - 1
            for t in T:-1:1
                path[t] = (idx % num_branch[t]) + 1
                idx = div(idx, num_branch[t])
            end
            paths[p] = path
        end
        return paths
    end
    paths = all_sample_paths(data.num_branch)
    num_item = data.num_item

    model = Model(Gurobi.Optimizer)
    set_time_limit_sec(model, time_limit)
    T = length(data.num_branch)
    num_paths = length(paths)
    if int_quantity_flag == 1
        @variable(model, 0 <= s[1:num_paths, t in 1:T, i in 1:num_item] <= data.inventory_capacity[t][i], Int)
        @variable(model, 0 <= b[1:num_paths, t in 1:T, i in 1:num_item] <= data.inventory_capacity[t][i], Int)
        @variable(model, 0 <= x[1:num_paths, t in 1:T, i in 1:num_item] <= data.production_capacity, Int)
    else
        @variable(model, 0 <= s[1:num_paths, t in 1:T, i in 1:num_item] <= data.inventory_capacity[t][i])
        @variable(model, 0 <= b[1:num_paths, t in 1:T, i in 1:num_item] <= data.inventory_capacity[t][i])
        @variable(model, 0 <= x[1:num_paths, t in 1:T, i in 1:num_item] <= data.production_capacity)
    end
    @variable(model, y[1:num_paths, 1:T, 1:num_item], Bin)

    for (n, path) in enumerate(paths)
        for t in 1:T
            @constraint(model, [i in 1:num_item], x[n, t, i] <= data.production_capacity * y[n, t, i])
            @constraint(model, sum(x[n, t, i] for i in 1:num_item) <= data.production_capacity)
            if t == 1
                @constraint(model, [i in 1:num_item], b[n, t, i] - s[n, t, i] + x[n, t, i] == data.demand[t][i][path[t]])
            else
                @constraint(model, [i in 1:num_item], b[n, t, i] - s[n, t, i] + x[n, t, i] == data.demand[t][i][path[t]] + b[n, t-1, i] - s[n, t-1, i])
            end
        end
    end

    # Non-anticipativity constraints
    for t in 1:T
        node_dict = Dict{Tuple{Vararg{Int}}, Int}()
        for (n, path) in enumerate(paths)
            node_key = Tuple(path[1:t])
            if haskey(node_dict, node_key)
                @constraint(model, [i in 1:num_item], s[node_dict[node_key], t, i] == s[n, t, i])
                @constraint(model, [i in 1:num_item], b[node_dict[node_key], t, i] == b[n, t, i])
                @constraint(model, [i in 1:num_item], x[node_dict[node_key], t, i] == x[n, t, i])
                @constraint(model, [i in 1:num_item], y[node_dict[node_key], t, i] == y[n, t, i])
            else
                node_dict[node_key] = n
            end
        end
    end

    p_n = 1.0 / num_paths
    @objective(model, Min, 
        sum(p_n * (
            sum(
                data.production_cost[t][i] * x[n, t, i] +
                data.fixed_cost[i] * y[n, t, i] +
                data.holding_cost[t][i] * s[n, t, i] +
                data.backlogging_cost[t][i] * b[n, t, i]
                for i in 1:num_item, t in 1:T
            )
        ) for n in 1:num_paths)
    )
    
    return Dict(
        :model => model,
        :x => x,
        :y => y,
        :s => s,
        :b => b,
        :paths => paths,
    )
end

function run_def(
    data::NamedTuple, 
    time_limit::Float64,
    int_quantity_flag::Int,
    def_presolve::Int,
    print_solution::Bool,
)
    def_model = build_def(data, time_limit, int_quantity_flag)
    if def_presolve == 0
        set_attribute(def_model[:model], "Presolve", 0)
        @info "DEF Presolve disabled."
    end
    optimize!(def_model[:model])
    if JuMP.is_solved_and_feasible(def_model[:model])
        @info "\nDEF Best Obj: $(objective_value(def_model[:model]))" * "\nDEF Best Bnd: $(objective_bound(def_model[:model]))"
        if print_solution
            @info "DEF solution" 
            println("x = $(JuMP.value.(def_model[:x]) |> x -> round.(x, digits=4))")
            println("y = $(JuMP.value.(def_model[:y]) |> x -> round.(x, digits=4))")
            println("s = $(JuMP.value.(def_model[:s]) |> x -> round.(x, digits=4))")
            println("b = $(JuMP.value.(def_model[:b]) |> x -> round.(x, digits=4))")
        end
    end
end

function main()
    # Parameters
    params = Parser.parse_args(additional_req_keys=Set([:num_item, :binarized_state_flag, :int_quantity_flag]))

    # data
    data = generate_data(params[:num_branch], params[:num_item], params[:int_quantity_flag]; seed=params[:seed], round_digits=2)
    # println(data)

    if params[:method] == 1
        scnbd_params = (
            r = params[:r],
            theta_bound = 0.0,
            time_limit = params[:time_limit],
            iteration_limit = haskey(params, :iteration_limit) ? params[:iteration_limit] : typemax(Int),
            cg_type = haskey(params, :cg_type) ? (
                params[:cg_type] == 1 ? ScaledCutNBD.CutGenerator :
                params[:cg_type] == 2 ? ScaledCutNBD.AcceleratedCutGenerator :
                params[:cg_type] == 3 ? ScaledCutNBD.LagrangianCutGenerator :
                params[:cg_type] == 4 ? ScaledCutNBD.BendersCutGenerator :
                error("Invalid cg_type: $(params[:cg_type]). Must be 1, 2, 3, or 4.")
            ) : ScaledCutNBD.CutGenerator,
            cg_bound = haskey(params, :cg_bound) ? params[:cg_bound] : 1e4,
            cgsp_tol = haskey(params, :cgsp_tol) ? params[:cgsp_tol] : 1e-4,
            fixed_iter_tol = haskey(params, :fixed_iter_tol) ? params[:fixed_iter_tol] : 1e-4,
            regularization_param = haskey(params, :regularization_param) ? params[:regularization_param] : 1e-6,
            cg_acc_until = haskey(params, :cg_acc_until) ? params[:cg_acc_until] : 100,
            cg_acc_level = haskey(params, :cg_acc_level) ? params[:cg_acc_level] : 0,
            cut_io_mode = haskey(params, :cut_io_mode) ? params[:cut_io_mode] : :n,
            cut_read_filename = haskey(params, :cut_read_filename) ? params[:cut_read_filename] : nothing,
            cut_write_filename = haskey(params, :cut_write_filename) ? params[:cut_write_filename] : nothing,
            print_timer_output = haskey(params, :print_timer) ? params[:print_timer] == 1 : false,
            log_file = nothing,
            binarized_state_flag = params[:binarized_state_flag],
            decimal_factor = haskey(params, :decimal_factor) ? params[:decimal_factor] : 3,
            int_quantity_flag = params[:int_quantity_flag],
        )
        run_scnbd(
            data,
            scnbd_params,
            false, # print_all_node_models
        )
    elseif params[:method] == 2
        run_def(
            data,
            params[:time_limit],
            params[:int_quantity_flag],
            haskey(params, :def_presolve) ? params[:def_presolve] : 1,
            false, # print_solution
        )
    else
        error("Method $(params[:method]) not supported.")
    end
end

main()