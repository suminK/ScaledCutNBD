#=
    CutGenerator for generating cuts for vₙ at node n.
    - Accelerated version using loose cuts (τ=0, i.e., LagrangianCut) for early iterations.
=#
mutable struct AcceleratedCutGenerator <: AbstractCutGenerator
    path::Tuple{Vararg{Int}}
    cgmp::Union{Nothing, Dict{Symbol, Any}}
    cgsp::Union{Nothing, Dict{Symbol, Any}}
end

function initialize(::Type{AcceleratedCutGenerator}, path::Tuple{Vararg{Int}})::AcceleratedCutGenerator
    return AcceleratedCutGenerator(path, nothing, nothing)
end

function build_cut_generators(::Type{AcceleratedCutGenerator}, tree::Tree, options::Options)
    build_cut_generators(CutGenerator, tree, options)
end

function generate_cut(
    ::Type{AcceleratedCutGenerator},
    forward_states::Dict{Symbol, Float64}, 
    approx_value::Float64, 
    node::Node,
    model_info::ModelInfo,
    options::Options,
)
    if length(node.ext[:nbd].runstats.iterations) <= options.cg_acc_until
        if options.cg_acc_level == 0
            # use Benders cuts for early iterations
            return generate_cut(
                BendersCutGenerator,
                forward_states,
                approx_value,
                node,
                model_info,
                options,
            )
        elseif options.cg_acc_level == 1
            return _generate_cut(
                forward_states,
            ) do 
                is_scaled_cut = false
                is_extended_space = true
                fixed_point_algorithm(
                    node.children .|> child -> build_cgmp(child, model_info, options, is_scaled_cut, is_extended_space),
                    node.children .|> child -> child.cut_generator.cgsp,
                    forward_states,
                    approx_value,
                    node,
                    options,
                )
            end
        elseif options.cg_acc_level == 2
            # use Lagrangian cuts for early iterations
            return generate_cut(
                LagrangianCutGenerator,
                forward_states,
                approx_value,
                node,
                model_info,
                options,
            )
        else
            error("Unknown cg_acc_level=$(options.cg_acc_level).")
        end
    end
    # use regular cuts for later iterations
    return _generate_cut(
        forward_states,
    ) do 
        is_scaled_cut = true
        is_extended_space = true
        fixed_point_algorithm(
            node.children .|> child -> build_cgmp(child, model_info, options, is_scaled_cut, is_extended_space),
            node.children .|> child -> child.cut_generator.cgsp,
            forward_states,
            approx_value,
            node,
            options,
        )
    end
end
