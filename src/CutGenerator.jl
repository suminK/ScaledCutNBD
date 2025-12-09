#=
    CutGenerator for generating cuts for vₙ at node n.
    IMPORTANT NOTE:
    * Note that the CutGenerator generates cuts for vₙ not Qₙ.
    * Generating cut requires both φₙ and φₐ₍ₙ₎.
    # TODO: support/check :Max stage_objective_sense case
=#
mutable struct CutGenerator <: AbstractCutGenerator
    path::Tuple{Vararg{Int}}
    # Dict that contains model, variables, and functions for CGMP
    cgmp::Union{Nothing, Dict{Symbol, Any}}
    # Dict that contains model, variables, and functions for CGSP
    cgsp::Union{Nothing, Dict{Symbol, Any}}
end

function initialize(::Type{CutGenerator}, path::Tuple{Vararg{Int}})::CutGenerator
    return CutGenerator(path, nothing, nothing)
end

function build_cut_generators(::Type{CutGenerator}, tree::Tree, options::Options)
    # All Convex Approxs must be built before building CGSPs.
    for (_, node) in tree.nodes
        if node.parent === nothing
            continue
        end
        if node.cut_generator.cgsp !== nothing
            continue
        end
        node.cut_generator.cgsp = build_cgsp(node, tree.model_info, options, tree.model_info.r)
    end
end

"""
    Build the CGMP for a given node.
    - `node`: Current node for which to build the CGMP.
    - `model_info`: ModelInfo object containing model building functions and parameters.
    - `options`: Options object containing algorithm parameters.
    - `is_scaled_cut`: Bool indicating whether to generate scaled cuts or not. (τ = 0 if false)
    - `is_extended_space`: Bool indicating whether to use extended space variables. (β[ext_space_vars] = 0 if false)
    Returns a Dict with the CGMP model and related functions.

    * Let's the current node be m.
    * The CGMP at node m:
        max (αₘ - βₘ' x_{[aʳ⁻¹(n):n]} - ρ (1 + τₘ))
        s.t. (αₘ, βₘ, τₘ) ∈ Π̂ʳₘ(ϕ).
      Here, n = a(n). So, x_{[aʳ⁻¹(n):n]} = x_{[aʳ(m):a(m)]}.

    * Cut is generated in the following form:
        Qₙ ≥ ∑ₘ qₙₘ (αₘ - βₘ' x_{[aʳ⁻¹(n):n]}) / (1 + ∑ₘ τₘ).
"""
function build_cgmp(
    node::Node, # Node m
    model_info::ModelInfo,
    options::Options,
    is_scaled_cut::Bool,
    is_extended_space::Bool,
)::Dict{Symbol, Any}
    parent_sub = node.parent.subproblem
    state_var_names = keys(parent_sub.state_vars) # x_a(m) = x_n
    extend_var_names = keys(parent_sub.extend_vars) # x_[aʳ(m):a²(m)] = x_{[a^{r-1}(n):a(n)]}
    all_var_names = union(state_var_names, extend_var_names)

    model = Model(model_info.optimizer_fn)
    set_silent(model)
    set_time_limit_sec(model, options.cg_time_limit)
    @variable(model, -options.cg_bound <= α <= options.cg_bound)
    @variable(model, -options.cg_bound <= β[all_var_names] <= options.cg_bound)
    @variable(model, 0 <= τ <= options.cg_bound)

    if !is_scaled_cut
        fix(τ, 0.0; force = true)
    end

    if !is_extended_space
        for k in extend_var_names
            fix(β[k], 0.0; force = true)
        end
    end

    # The regularization term for numerical stability
    regularization = @expression(model, options.regularization_param * (τ^2 + sum(β[k]^2 for k in all_var_names)))
    # regularization = @expression(model, options.regularization_param * (sum(β[k]^2 for k in all_var_names)))

    """
        Add a constraint in this form: cₘ'x̂ₘ + θ̂ₘ + βₘ'x̂_{[a^{r-1}(n):n]} + τₘ * θ̂ₙ >= αₘ
    """
    function add_constraint(solution_dict::Dict{Symbol, Any})
        stage_objective_value = solution_dict[:stage_objective_value] # cₘ'x̂ₘ
        theta_m_value = solution_dict[:theta_m_value] # θ̂ₘ
        ext_state_values = solution_dict[:ext_state_values] # x̂_{[a^{r-1}(n):n]}
        theta_n_value = solution_dict[:theta_n_value] # θ̂ₙ
        @constraint(model, stage_objective_value + theta_m_value + sum(β[k] * ext_state_values[k] for k in β.axes[1]) + τ * theta_n_value >= α)
    end

    function parameterize(ext_state_values::Dict{Symbol, Float64}, ρ::Float64)
        @objective(
            model, Max, 
            α - ρ * (1 + τ) - sum(β[k] * ext_state_values[k] for k in all_var_names) - regularization
        )
    end
    
    return Dict(
        :model => model,
        :α => α,
        :β => β,
        :τ => τ,
        :parameterize => parameterize,
        :add_constraint => add_constraint,
        :regularization => regularization,
    )
end

"""
    Build the CGSP for a given node.
    - `node`: Current node for which to build the CGSP.
    - `model_info`: ModelInfo object containing model building functions and parameters.
    - `options`: Options object containing algorithm parameters.
    - `r`: depth parameter.
    Returns a Dict with the CGSP model and related functions.

    * The CGSP at node m: (for generating cut for vₘ)
        min cₘ'xₘ + φₘ(x_{[aʳ⁻¹(m):m]}) + β̂ₘ'x_{[aʳ⁻¹(n):n]} + τ̂ₘ * φₙ(x_{[aʳ⁻¹(n):n]}) - α̂ₘ
        s.t. subproblem constraints at node m:
            T_{m} x_{a(m)} + W_{m} x_{m} = h_{m}
            T_{a(m)} x_{a^2(m)} + W_{a(m)} x_{a(m)} = h_{a(m)}
            ...
            T_{a^{r-1}(m)} x_{a^{r}(m)} + W_{a^{r-1}(m)} x_{a^{r-1}(m)} = h_{a^{r-1}(m)}
            (so, r constraint sets in total.),
            and
            x_m ∈ Xₘ, x_{a(m)} ∈ X_{a(m)}, ..., x_{a^{r}(m)} ∈ X_{a^{r}(m)} (simple bound/integrality sets).
        With the stagewise independence, it is equivalent to define the constraints over
            x_tₘ, x_{tₘ-1}, ..., x_{tₘ-r}.
    * Note that x_{a^r(m)} is constrained by only the simple bound/integrality set X_{tₘ-r}. (We may restrict this further?)
"""
function build_cgsp(
    node::Node,
    model_info::ModelInfo,
    options::Options,
    r::Int,
)::Dict{Symbol, Any}
    is_leaf = node.depth == model_info.num_stage
    t = node.depth # tₘ

    # Create the CGSP model
    model = Model(model_info.optimizer_fn)
    set_silent(model)
    # Set MIP gap to 0 for exact solution
    # set_optimizer_attribute(model, "MIPGap", 1e-6)
    set_time_limit_sec(model, options.cg_time_limit)
    # Define theta variable (θₘ >= φₘ and θₙ >= φₙ)
    theta_m = @variable(model, theta_m)
    theta_n = @variable(model, theta_n)
    if is_leaf
        @constraint(model, theta_m == 0.)
    else
        set_lower_bound(theta_m, model_info.theta_bound)
    end
    set_lower_bound(theta_n, model_info.theta_bound)
    model.ext[:theta_m] = theta_m
    model.ext[:theta_n] = theta_n

    model.ext[:state_vars] = Dict{Symbol, State}()
    model.ext[:another_object_dictionary] = Dict{Symbol, Any}()
    original_state_var_names::Vector{Symbol} = [state.original_name for (_, state) in node.subproblem.state_vars]
    opt_sense = nothing
    stage_objective = nothing
    last_stage_parameterize = nothing
    for lag in 0:(r - 1)
        if t - lag >= 1
            """
                Build the CGSP using subproblem_builder.
                It copies the model structure for t ∈ {tₘ, tₘ-1, ..., tₘ-(r-1)}.
                It copies `State` variables for all the stages, so we need to connect their dependencies manually here.
            """
            tmp_opt_sense, tmp_stage_objective, tmp_parameterize = model_info.builder(model, t - lag)
            tmp_parameterize(t - lag, node.path[t - lag])
            if lag == 0
                last_stage_parameterize = tmp_parameterize
            end
            # keep the objective function.
            if lag == 0 
                opt_sense = tmp_opt_sense
                stage_objective = tmp_stage_objective
            end
            """
                Connect the dependencies of the state variables.
                The name pattern of the extended state variables are: "`original_name`_`stage`".
                That is, x_t.in = x_{t-1}.out.
            """
            if lag > 0
                for name in original_state_var_names
                    this_stage_name = Symbol(name, "_$(t - lag)")
                    prev_stage_name = Symbol(name, "_$(t - lag + 1)")
                    @constraint(model, model.ext[:state_vars][this_stage_name].out == model.ext[:state_vars][prev_stage_name].in)
                end
            end
        end
    end

    #=
        Prepare the state_vars_by_name dict to refer the state variables in CGSP by their names.
        - names: x_t, ..., x_{t-(r-1)}, x_{t-r}
        - in formulation: x_m, ..., x_{a^{r-1}(m)}, x_{a^{r}(m)}
        - objects: x_t.out, ..., x_{t-(r-1)}.out, **x_{t-(r-1)}.in**
    =#
    model.ext[:state_vars_by_name] = Dict{Symbol, JuMP.VariableRef}()
    for (state_name, state) in model.ext[:state_vars]
        model.ext[:state_vars_by_name][state_name] = state.out
    end
    # Get the earliest generated stage. (t_e)
    earliest_gen_stage = max(t - (r - 1), 1)
    model.ext[:earliest_gen_stage] = earliest_gen_stage
    # The state variables at t_e - 1 are defined as the incoming state variables at stage t_e.
    for name in original_state_var_names
        tmp_name = Symbol(name, "_$(earliest_gen_stage)")
        in_state = model.ext[:state_vars][tmp_name]
        incoming_state_var_name = Symbol(name, "_$(earliest_gen_stage - 1)")
        model.ext[:state_vars_by_name][incoming_state_var_name] = in_state.in
        
        #=
          If t_e == 1, fix the incoming state variables to initial values. 
          (This is automatically handled in JuMP.jl when creating the state variables.)
          If not, set the same bounds and integrality constraint on the incoming state variables.
        =#
        if earliest_gen_stage > 1
            if is_binary(in_state.out)
                set_binary(in_state.in)
                # in this case, relax_integrality automatically adds bounds [0, 1].
                continue
            end
            # If not binary, the bounds must be explicitly set.
            if !(has_lower_bound(in_state.out) && has_upper_bound(in_state.out))
                error("The state variable must be bounded. Variable name: $(in_state), is_integer=$(is_integer(in_state.out))")
            end
            set_lower_bound(in_state.in, lower_bound(in_state.out))
            set_upper_bound(in_state.in, upper_bound(in_state.out))
            if is_integer(in_state.out)
                set_integer(in_state.in)
            end
        end
    end

    #=
        Define the add_cut functions for theta_m and theta_n. 
        The cuts extended spaces are as follows. The input slope must be defined over the corresponding state variables.
        
        [theta_m]
        space: x_[a^{r-1}(m):m], tₘ - r + 1 to tₘ.
        theta_m_state_names = [Symbol(name, "_$(τ)") for name in original_state_var_names, τ in max(t - r + 1, 1):t]
        [theta_n]
        space: x_[a^{r}(m):a(m)], tₘ - r to tₘ - 1.
        theta_n_state_names = [Symbol(name, "_$(τ)") for name in original_state_var_names, τ in max(t - r, 1):(t-1)]
    =#
    function add_cut_fn(theta_var::VariableRef, slope::Dict{Symbol, Float64}, intercept::Float64)
        @constraint(model, theta_var >= intercept + sum(val * model.ext[:state_vars_by_name][k] for (k, val) in slope))
    end
    if !is_leaf
        push!(node.convex_approx.model_hooks, ModelHook(model, (slope::Dict{Symbol, Float64}, intercept::Float64) -> add_cut_fn(model.ext[:theta_m], slope, intercept)))
    end
    push!(node.parent.convex_approx.model_hooks, ModelHook(model, (slope::Dict{Symbol, Float64}, intercept::Float64) -> add_cut_fn(model.ext[:theta_n], slope, intercept)))

    """
        Parameterize the CGSP with (α, β, τ) from CGMP.
        - `α`, `β`, `τ`: αₘ, βₘ, τₘ from CGMP.
        - `i`: branch index ∈ num_branch[t].
    """
    function parameterize(α::Float64, β::Dict{Symbol, Float64}, τ::Float64, i::Int)
        # Parameterize the objective function
        if opt_sense == MIN_SENSE
            last_stage_parameterize(t, i)
            @objective(model, Min, 
                stage_objective + model.ext[:theta_m] + τ * model.ext[:theta_n] - α + sum(β[k] * model.ext[:state_vars_by_name][k] for k in keys(β))
            )
        else
            error("Unsupported optimize_sense in CGSP: $opt_sense")
        end
    end

    function get_solution()::Dict{Symbol, Any}
        ext_state_values = Dict{Symbol, Float64}()
        for (name, var) in model.ext[:state_vars_by_name]
            ext_state_values[name] = JuMP.value(var)
        end
        stage_objective_value = JuMP.value(stage_objective)
        theta_m_value = JuMP.value(model.ext[:theta_m])
        theta_n_value = JuMP.value(model.ext[:theta_n])
        return Dict(
            :ext_state_values => ext_state_values,
            :stage_objective_value => stage_objective_value,
            :theta_m_value => theta_m_value,
            :theta_n_value => theta_n_value,
        )
    end

    return Dict(
        :model => model,
        :get_solution => get_solution,
        :parameterize => parameterize,
    )
end


function generate_cut(
    ::Type{CutGenerator},
    forward_states::Dict{Symbol, Float64},
    approx_value::Float64,
    node::Node,
    model_info::ModelInfo,
    options::Options,
)
    return _generate_cut(
        forward_states,
    ) do 
        fixed_point_algorithm(
            node.children .|> child -> build_cgmp(child, model_info, options, true, true),
            node.children .|> child -> child.cut_generator.cgsp,
            forward_states,
            approx_value,
            node,
            options,
        ) 
    end
end


"""
    Generate a cut for Qₙ at node n.
    - `forward_states`: Dict of extended state variable values, over which the outer approximation is defined.
    - `approx_value`: The approximate value at the current node.
    - `node`: Current node for which to generate the cut.
    - `model_info`: ModelInfo object containing model building functions and parameters.
    - `options`: Options object containing algorithm parameters.
    Returns a Dict with the cut parameters: slope and intercept.
"""
function _generate_cut(
    run_cg_algorithm::Function,
    forward_states::Dict{Symbol, Float64}, 
)::Dict{Symbol, Any}
    solution_info = run_cg_algorithm()
    slope = Dict{Symbol, Float64}()
    for (k, v) in solution_info[:β]
        slope[k] = - v / (1 + solution_info[:τ])
    end
    intercept = solution_info[:α] / (1 + solution_info[:τ])
    return Dict(
        :slope => slope,
        :intercept => intercept,
        :α => solution_info[:α],
        :β => solution_info[:β],
        :τ => solution_info[:τ],
        :evaluated_obj => intercept + sum(slope[k] * forward_states[k] for k in keys(slope)),
    )
end

"""
    Fixed-point iteration algorithm for generating cuts.
    - `cgmps`: Vector of CGMP dicts for each child node.
    - `cgsps`: Vector of CGSP dicts for each child node.
    - `forward_states`: Dict of forward-pass state values at the current node.
    - `ρ_0`: Initial guess for ρ.
    - `node`: Current node for which to generate cuts.
    - `options`: Options object containing algorithm parameters.
    Returns a Dict with solution information.
"""
function fixed_point_algorithm(
    cgmps::Vector{Dict{Symbol, Any}},
    cgsps::Vector{Dict{Symbol, Any}},
    forward_states::Dict{Symbol, Float64}, 
    ρ_0::Float64, 
    node::Node,
    options::Options,
)::Dict{Symbol, Any}
    N = length(node.children)
    β_keys = cgmps[1][:β].axes[1]
    ρ_k = ρ_0
    iter_counter = 0
    total_inner_iter_counter = 0
    while true
        iter_counter += 1
        cgp_solution = solve_cgp(cgmps, cgsps, forward_states, ρ_k, options.cgsp_tol, options.cg_inner_iteration_limit)
        
        # Compute the means
        round_fn = x -> round(x, digits=10)
        α_mean = sum(cgp_solution[:α]) / N .|> round_fn
        β_mean = Dict{Symbol, Float64}(k => (sum(cgp_solution[:β][i][k] for i in 1:N) / N) .|> round_fn for k in β_keys) 
        τ_mean = sum(cgp_solution[:τ]) / N .|> round_fn
        total_inner_iter_counter += cgp_solution[:total_iterations]

        #=
            Compute C_ρ
            Note: the ρ_k may result in negative C_ρ, as the corresponding solution (α_mean, β_mean, τ_mean) could be non-optimal (and even infeasible depending on solvers).
        =#
        β_x_term = sum(β_mean[k] * forward_states[k] for k in β_keys)
        C_ρ = α_mean - β_x_term - ρ_k * (1 + τ_mean)
        if C_ρ < options.fixed_iter_tol
            return Dict(
                :C_ρ => C_ρ,
                :ρ => ρ_k,
                :α => α_mean,
                :β => β_mean,
                :τ => τ_mean,
                :iter_counter => iter_counter,
            )
        else
            ρ_k += C_ρ / (1 + τ_mean)
        end
    end
end

function solve_cgp(
    cgmps::Vector{Dict{Symbol, Any}},
    cgsps::Vector{Dict{Symbol, Any}},
    forward_states::Dict{Symbol, Float64},
    ρ_k::Float64,
    cgsp_tol::Float64,
    cg_inner_iteration_limit::Int,
)::Dict{Symbol, Any}
    N = length(cgmps)
    β_keys = cgmps[1][:β].axes[1]
    # Parameterize CGMPs with forward_states and ρ_k
    for i in 1:N
        cgmps[i][:parameterize](forward_states, ρ_k)
    end
    # solve C(ρ_k)
    α = zeros(N)
    β = [Dict{Symbol, Float64}() for _ in 1:N]
    τ = zeros(N)
    total_inner_iterations = 0
    for i in 1:N
        prev_cgsp_obj = nothing
        tmp_inner_iteration_counter = 0
        while true
            tmp_inner_iteration_counter += 1
            # Solve CGMP
            @timeit timer_output "CGMP_solve" begin
                solve!(cgmps[i][:model])
            end
            round_fn = x -> round(x, digits=10)
            α[i] = value(cgmps[i][:α]) |> round_fn
            for k in β_keys
                β[i][k] = value(cgmps[i][:β][k]) |> round_fn
            end
            τ[i] = value(cgmps[i][:τ]) |> round_fn
            # Solve CGSP
            cgsps[i][:parameterize](α[i], β[i], τ[i], i)
            @timeit timer_output "CGSP_solve" begin
                solve!(cgsps[i][:model])
            end
            cgsp_obj = objective_value(cgsps[i][:model])
            cgsp_obj_ = cgsp_obj + α[i]

            # Check termination
            if cgsp_obj >= - cgsp_tol
                break
            end
            if cgsp_obj >= - min(cgsp_tol * 1e2, 1e-1) && prev_cgsp_obj !== nothing && isapprox(cgsp_obj, prev_cgsp_obj, atol=cgsp_tol * 1e-2)
                # Project down α[i] for feasibility
                α[i] = cgsp_obj_
                break
            end
            if tmp_inner_iteration_counter >= cg_inner_iteration_limit
                @info "CGMP-CGSP loop reached the maximum inner iteration limit. This might indicate numerical issues for solving the CGSP with the current solver." max_inner_iterations=cg_inner_iteration_limit CGSP_obj=round(cgsp_obj, digits=6) previous_obj=prev_cgsp_obj === nothing ? "N/A" : round(prev_cgsp_obj, digits=6)
                α[i] = cgsp_obj_
                break
            end
            # otherwise, continue the CGMP-CGSP loop
            prev_cgsp_obj = cgsp_obj
            # add a violated constraint to CGMP
            solution_dict = cgsps[i][:get_solution]()
            cgmps[i][:add_constraint](solution_dict)
        end
        total_inner_iterations += tmp_inner_iteration_counter
    end

    return Dict(
        :α => α,
        :β => β,
        :τ => τ,
        :total_iterations => total_inner_iterations,
    )
end