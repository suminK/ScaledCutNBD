function Node(
    path::Tuple{Vararg{Int}},
    depth::Int,
    parent::Union{Nothing, Node},
    subproblem::Subproblem,
    cut_generator::Union{Nothing, AbstractCutGenerator},
    convex_approx::Union{Nothing, ConvexApprox},
)::Node
    return Node(
        path, 
        depth, 
        parent, 
        Vector{Node}(), 
        subproblem, 
        cut_generator, 
        convex_approx,
        Dict{Symbol, Any}(), # ext
    )
end

function Base.show(io::IO, node::Node)
    print(io, 
        "(Node $(node.path): " * 
        "Depth=$(node.depth), " * 
        "#Children=$(length(node.children))" *
        ")"
    )
    return
end

"""
    Recursively add child nodes to the tree.
    - `nodes`: Dict to store all nodes.
    - `subproblems`: Dict to store all subproblems.
    - `cut_generators`: Dict to store all cut generators.
    - `parent`: The parent node to add children to.
    - `model_info`: ModelInfo struct containing model parameters.
    Updates `nodes`, `subproblems`, and `cut_generators` in place.
"""
function add_node!(
    nodes::Dict{Tuple{Vararg{Int}}, Node}, 
    subproblems::Dict{Tuple{Vararg{Int}}, Subproblem}, 
    cut_generators::Dict{Tuple{Vararg{Int}}, AbstractCutGenerator},
    convex_approxs::Dict{Tuple{Vararg{Int}}, ConvexApprox},
    parent::Node, 
    model_info::ModelInfo,
)
    parent_t = parent.depth
    current_t = parent_t + 1
    if parent_t <= model_info.num_stage - 1
        for m in 1:model_info.num_branch[current_t]
            child_path::Tuple{Vararg{Int}} = (parent.path..., m) # (1, ..., n, m)
            is_leaf_node = current_t == model_info.num_stage
            # Subproblem path = Path(φₘ). If leaf node, Path(0ₘ), i.e., all zeros.
            subproblem_path::Tuple{Vararg{Int}} = is_leaf_node ? Tuple(zeros(Int, model_info.num_stage)) : get_zeroed_out_path(child_path, model_info.r)
            # CutGenerator path = (overlap of Path(φₙ) and Path(φₘ). If leaf node, Path(φₙ, 0).
            cut_generator_path::Tuple{Vararg{Int}} = (parent.convex_approx.path..., subproblem_path[end])
            #= 
                Create ConvexApprox φₙ (<= Qₙ = ∑ₘ qₙₘ vₘ).
                IMPORTANT NOTE:
                The cut for vₘ depends on ξ_[tₘ - r + 1:tₘ], and so does φₙ path.
                e.g) tₙ=3, r=2. (1, 2, 2) -> (0, 0, 2), because a cut for vₘ depends on ξ₃ and ξ₄.
                e.g) tₙ=3, r=1. (1, 2, 2) -> (0, 0, 0), because a cut for vₘ depends on ξ₄.
            =#
            φ = nothing
            if !is_leaf_node
                if !haskey(convex_approxs, subproblem_path)
                    convex_approxs[subproblem_path] = ConvexApprox(current_t, subproblem_path, Vector{ScaledCut}(), Vector{ModelHook}())
                end
                φ = convex_approxs[subproblem_path]
            end
            # Create subproblem 
            if !haskey(subproblems, subproblem_path)
                subproblems[subproblem_path] = Subproblem(subproblem_path, φ, model_info, is_leaf_node)
            end
            # Create dummy cut_generator; will be populated later
            cut_generator = nothing
            if !haskey(cut_generators, cut_generator_path)
                cut_generator = initialize(model_info.cut_generator_type, cut_generator_path)
                cut_generators[cut_generator_path] = cut_generator
            end
            # Build child node
            child_node = Node(
                child_path,
                current_t,
                parent,
                subproblems[subproblem_path],
                cut_generators[cut_generator_path],
                φ,
            )
            push!(parent.children, child_node)
            nodes[child_path] = child_node
            add_node!(nodes, subproblems, cut_generators, convex_approxs, child_node, model_info)
        end
    end
end

"""
    Construct a Tree.
    - `builder`: A function to build subproblem.
    - `num_branch`: A vector indicating the number of branches (realizations) at each stage.
    - `r`: The depth for the extended space over which the outer approximation is defined. 1={t}, 2={t, t-1}, 3={t, t-1, t-2}, ...
    - `optimizer_fn`: Optimizer constructor.
    - `theta_bound`: Bound for θ variable in subproblems.
    - `cut_generator_type`: Type of cut generator to use (default: CutGenerator).
"""
function Tree(
    builder::Function, 
    num_branch::Vector{Int}, 
    r::Int, 
    optimizer_fn::Function, 
    theta_bound::Float64,
    ;
    cut_generator_type::Type{<:AbstractCutGenerator} = CutGenerator,
)::Tree
    # Argument validity checks
    if length(num_branch) < 2
        throw(ArgumentError("num_branch must have at least two stages."))
    end
    if (1 <= r <= length(num_branch)) == false
        throw(ArgumentError("r must satisfy 1 <= r <= T"))
    end
    if num_branch[1] != 1
        throw(ArgumentError("The first element of num_branch must be 1 (the root node)."))
    end

    model_info = ModelInfo(
        length(num_branch),
        num_branch,
        r,
        theta_bound,
        optimizer_fn,
        extend_builder(builder),
        cut_generator_type,
    )

    """
        Initialize containers for the tree structure.
        * We create all nodes in the tree.
        * We create cut_generators for all nodes, except the root node.
        * We create only required amount of subproblems and convex_approxs, referred by their zeroed-out paths.
    """
    nodes = Dict{Tuple{Vararg{Int}}, Node}() # n
    subproblems = Dict{Tuple{Vararg{Int}}, Subproblem}() # v̂ₙ
    cut_generators = Dict{Tuple{Vararg{Int}}, AbstractCutGenerator}() # CGMP and CGSP at m
    convex_approxs = Dict{Tuple{Vararg{Int}}, ConvexApprox}() # ϕₙ
    # Create root node
    root_path = (1,)
    convex_approxs[root_path] = ConvexApprox(1, root_path, Vector{ScaledCut}(), Vector{ModelHook}())
    subproblems[root_path] = Subproblem(root_path, convex_approxs[root_path], model_info, false)
    root_node = Node(root_path, 1, nothing, subproblems[root_path], nothing, convex_approxs[root_path])
    nodes[root_path] = root_node
    # Recursively add the other nodes
    add_node!(nodes, subproblems, cut_generators, convex_approxs, root_node, model_info)
    # Return the tree
    return Tree(root_node, nodes, subproblems, cut_generators, convex_approxs, model_info)
end

function extend_builder(builder::Function)::Function
    extended_builder = function(model::JuMP.Model, t::Int)
        # (1) set the current stage in model.ext.
        model.ext[:stage] = t # set the current stage in ext, for variable naming purposes.
        # (2) build the model.
        obj_sense, stage_obj_fn, parameterize_fn = builder(model, t)
        # (3) re-register all objects with updated names for the current stage t.
        for (sym, item) in object_dictionary(model)
            JuMP.unregister(model, sym)
            model.ext[:another_object_dictionary][Symbol(sym, "_$t")] = item
            if item isa State || item isa Array{State{T}} where T 
                # for State, the variable names are already handled in add_variable
                continue
            else
                item .|> e -> JuMP.set_name(e, JuMP.name(e) * "_$t")
            end
        end

        # Validity check
        if obj_sense ∉ [MIN_SENSE, MAX_SENSE]
            error("Objective sense must be either Min or Max.")
        end
        if stage_obj_fn isa JuMP.GenericAffExpr == false
            error("The builder function must return an affine objective function.")
        end
        if parameterize_fn isa Function == false
            error("The builder function must return a parameterize function.")
        end
        return obj_sense, stage_obj_fn, parameterize_fn
    end
    return extended_builder
end

function Base.show(io::IO, tree::Tree)
    print(io, 
        "(Tree: " *
        "#nodes=$(length(tree.nodes)), " *
        "#subproblems=$(length(tree.subproblems)), " *
        "#convex_approxs=$(length(tree.convex_approxs)), " *
        "#cut_generators=$(length(tree.cut_generators)), " *
        "#branches=$(tree.model_info.num_branch)), " *
        "root=$(tree.root_node)" *
        ")"
    )
end

function Base.show(io::IO, c::ConvexApprox)
    print(io, 
        "(ConvexApprox $(c.path): " * 
        "Stage=$(c.stage), " * 
        "#Cuts=$(length(c.cuts)), " *
        "#ModelHooks=$(length(c.model_hooks))" *
        ")"
    )
end

