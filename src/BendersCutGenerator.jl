#=
    Benders CutGenerator for generating cuts for vₙ at node n.
    - Benders cuts are generated for the state variables only at the current stage.
=#
mutable struct BendersCutGenerator <: AbstractCutGenerator
    path::Tuple{Vararg{Int}}
    cgmp::Union{Nothing, Dict{Symbol, Any}}
    cgsp::Union{Nothing, Dict{Symbol, Any}}
end

function initialize(::Type{BendersCutGenerator}, path::Tuple{Vararg{Int}})::BendersCutGenerator
    return BendersCutGenerator(path, nothing, nothing)
end

function build_cut_generators(::Type{BendersCutGenerator}, tree::Tree, options::Options)
    if tree.model_info.r > 1
        @warn "BendersCutGenerator, but got r=$(tree.model_info.r)."
    end
    build_cut_generators(CutGenerator, tree, options)
end

function generate_cut(
    ::Type{BendersCutGenerator},
    forward_states::Dict{Symbol, Float64}, 
    approx_value::Float64, 
    node::Node,
    model_info::ModelInfo,
    options::Options,
)
    return _generate_cut(
        forward_states,
    ) do 
        # get incoming state variable values
        in_state_values = Dict{Symbol, Float64}(
            state.original_name => forward_states[name] for (name, state) in node.subproblem.model.ext[:state_vars]
        )
        # solve LP relaxation of Subproblem and compute cut coefficients
        obj_exp = 0.0
        duals_exp = Dict{Symbol, Float64}(state.original_name => 0.0 for state in values(node.subproblem.model.ext[:state_vars]))
        N = length(node.children)
        for child in node.children
            child.subproblem.parameterize(child.path[end])
            # fix states to incoming state values and extend_vars to 0
            for (_, state) in child.subproblem.model.ext[:state_vars]
                fix(state.in, in_state_values[state.original_name])
            end
            for (_, var) in child.subproblem.model.ext[:extend_vars]
                fix(var, 0.0)
            end
            undo_relax = relax_integrality(child.subproblem.model)
            solve!(child.subproblem.model)
            for (_, state) in child.subproblem.model.ext[:state_vars]
                duals_exp[state.original_name] += (1/N) * JuMP.dual(JuMP.FixRef(state.in))
            end
            obj_exp += (1/N) * objective_value(child.subproblem.model)
            # undo relax_integrality
            undo_relax()
            # unfix variables
            for (_, state) in child.subproblem.model.ext[:state_vars]
                unfix(state.in)
            end
            for (_, var) in child.subproblem.model.ext[:extend_vars]
                unfix(var)
            end
        end

        # compute outputs
        round_fn = x -> round(x, digits=10)
        α = obj_exp - sum(duals_exp[k] * in_state_values[k] for k in keys(in_state_values)) |> round_fn
        β = Dict{Symbol, Float64}(key => - duals_exp[state.original_name] |> round_fn for (key, state) in node.subproblem.model.ext[:state_vars])

        return Dict{Symbol, Any}(
            :α => α,
            :β => β,
            :τ => 0.0,
        )
    end
end