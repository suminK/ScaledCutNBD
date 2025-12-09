#=
    Lagrangian CutGenerator for generating cuts for vₙ at node n.
=#
mutable struct LagrangianCutGenerator <: AbstractCutGenerator
    path::Tuple{Vararg{Int}}
    cgmp::Union{Nothing, Dict{Symbol, Any}}
    cgsp::Union{Nothing, Dict{Symbol, Any}}
end

function initialize(::Type{LagrangianCutGenerator}, path::Tuple{Vararg{Int}})::LagrangianCutGenerator
    return LagrangianCutGenerator(path, nothing, nothing)
end

function build_cut_generators(::Type{LagrangianCutGenerator}, tree::Tree, options::Options)
    if tree.model_info.r > 1
        @warn "LagrangianCutGenerator, but got r=$(tree.model_info.r)."
    end
    build_cut_generators(CutGenerator, tree, options)
end

function generate_cut(
    ::Type{LagrangianCutGenerator},
    forward_states::Dict{Symbol, Float64}, 
    approx_value::Float64, 
    node::Node,
    model_info::ModelInfo,
    options::Options,
)
    return _generate_cut(
        forward_states,
    ) do 
        cgp_solution = solve_cgp(
            node.children .|> child -> build_cgmp(child, model_info, options, false, false),
            node.children .|> child -> child.cut_generator.cgsp,
            forward_states,
            0.0,
            options.cgsp_tol,
            options.cg_inner_iteration_limit,
        )
        N = length(node.children)
        round_fn = x -> round(x, digits=10)
        return Dict{Symbol, Any}(
            :α => sum(cgp_solution[:α]) / N |> round_fn,
            :β => Dict{Symbol, Float64}(k => (sum(cgp_solution[:β][i][k] for i in 1:N) / N) |> round_fn for k in keys(cgp_solution[:β][1])),
            :τ => 0.0,
        )
    end
end