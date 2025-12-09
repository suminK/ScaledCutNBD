function Subproblem(
    path::Tuple{Vararg{Int}}, 
    convex_approx::Union{Nothing, ConvexApprox},
    model_info::ModelInfo,
    is_leaf_node::Bool,
)::Subproblem
    @assert is_leaf_node || (convex_approx !== nothing)

    t = length(path)
    # Create the subproblem model
    model = Model(model_info.optimizer_fn)
    set_silent(model)
    model.ext[:state_vars] = Dict{Symbol, State}()
    model.ext[:extend_vars] = Dict{Symbol, JuMP.VariableRef}()
    model.ext[:another_object_dictionary] = Dict{Symbol, Any}()
    # build the model
    opt_sense, stage_objective, parameterize_fn = model_info.builder(model, length(path)) 
    
    # initialize the extended space variables
    # for non-leaf nodes.
    if !is_leaf_node
        for lag in 1:(model_info.r - 1)
            if t - lag >= 1
                # copy state variables with added stage (t-lag) suffix.
                for (_, state) in model.ext[:state_vars]
                    var_name = "$(state.original_name)_$(t-lag)"
                    var = @variable(model, base_name = var_name)
                    add_var_ref!(model, Symbol(var_name), var, :extend)
                end
            end
        end
    end

    # Define objective function
    theta = @variable(model, theta)
    model.ext[:theta] = theta

    # if leaf node, fix theta to 0.0
    if is_leaf_node
        fix(theta, 0.0)
    else
        if opt_sense == MIN_SENSE
            set_lower_bound(theta, model_info.theta_bound)
        elseif opt_sense == MAX_SENSE
            set_upper_bound(theta, model_info.theta_bound)
        else
            # should not reach here, since already checked in `extend_builder`
            error("opt_sense must be either MIN_SENSE or MAX_SENSE.")
        end

        # Set the add_cut function for convex approximation
        function add_cut_func(slope::Dict{Symbol, Float64}, intercept::Float64)
            @constraint(model, 
                theta >= intercept 
                    + sum(val * model.ext[:state_vars][k].out for (k, val) in slope if haskey(model.ext[:state_vars], k)) 
                    + sum(val * model.ext[:extend_vars][k] for (k, val) in slope if haskey(model.ext[:extend_vars], k))
            )
        end
        push!(convex_approx.model_hooks, ModelHook(model, add_cut_func))
    end
    # Set objective
    if opt_sense == MIN_SENSE
        @objective(model, Min, stage_objective + theta)
    elseif opt_sense == MAX_SENSE
        @objective(model, Max, stage_objective + theta)
    else
        # should not reach here, since already checked in extend_builder
        error("opt_sense must be either MIN_SENSE or MAX_SENSE.")
    end

    # Set parameterize function
    parameterize = (n::Int) -> parameterize_fn(t, n)

    # Return a new Subproblem object
    return Subproblem(path, model, stage_objective, theta, model.ext[:state_vars], model.ext[:extend_vars], parameterize)
end

function Base.show(io::IO, sub::Subproblem)
    print(io, 
        "(Stage subproblem: " * 
        "path=$(sub.path), " *
        "state_vars=$(sub.state_vars), " *
        "extend_vars=$(sub.extend_vars)" *
        ")"
    )
    return
end

function add_var_ref!(model::Model, sym_name::Symbol, var::Union{State, JuMP.VariableRef}, var_type::Symbol)
    @assert var_type in (:state, :extend)
    var_dict = nothing
    if var_type == :state
        var_dict = model.ext[:state_vars]
    elseif var_type == :extend
        var_dict = model.ext[:extend_vars]
    end
    if haskey(var_dict, sym_name)
        error("A variable with the same sym_name ($sym_name) already exists in the following model:\n$(model)\nThe var_dict is $(var_dict).")
    end
    var_dict[sym_name] = var
end

"""
    Get the outgoing state values from a subproblem after it has been solved.
    Returns a Dict with outgoing values.
"""
function get_outgoing_state(subproblem::Subproblem)
    values = Dict{Symbol,Float64}()
    for (name, state) in subproblem.state_vars
        outgoing_value = JuMP.value(state.out)
        values[name] = outgoing_value
    end
    return values
end