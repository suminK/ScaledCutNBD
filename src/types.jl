abstract type AbstractCutGenerator end

struct ModelInfo
    num_stage::Int
    num_branch::Vector{Int}
    r::Int
    theta_bound::Float64
    optimizer_fn::Function
    builder::Function
    cut_generator_type::Type{<:AbstractCutGenerator}
end

struct State{T}
    in::T
    out::T
    original_name::Symbol
end

struct StateInfo
    in::JuMP.VariableInfo
    out::JuMP.VariableInfo
    initial_value::Float64
    kwargs::Any
end

struct ScaledCut
    α::Float64
    β::Dict{Symbol, Float64}
    τ::Float64
end

struct ModelHook
    model::JuMP.Model
    add_cut_func::Function
end

struct ConvexApprox
    stage::Int
    path::Tuple{Vararg{Int}}
    cuts::Vector{ScaledCut}
    model_hooks::Vector{ModelHook} # When a new cut is added, these models need to be updated using `add_cut_func`.
end

struct Subproblem
    # Path in the scenario tree associated with the outer approximation and `r`.
    path::Tuple{Vararg{Int}}
    # Stage subproblem
    model::JuMP.Model
    stage_objective::JuMP.GenericAffExpr{Float64,JuMP.VariableRef}
    theta::Union{Nothing, JuMP.VariableRef}
    state_vars::Dict{Symbol, State}
    extend_vars::Dict{Symbol, JuMP.VariableRef}
    parameterize::Function # parameterize(branch_idx::Int)
    #
end

struct Node
    path::Tuple{Vararg{Int}}
    depth::Int
    parent::Union{Nothing, Node}
    children::Vector{Node}
    subproblem::Subproblem
    cut_generator::Union{Nothing, AbstractCutGenerator}
    convex_approx::Union{Nothing, ConvexApprox}
    ext::Dict{Symbol, Any} # For any extra information to be stored
end

"""
    Tree structure for the scenario tree.
    - `root_node`: The root node of the tree.
    - `nodes`: A dictionary mapping node paths to Node objects.
    - `subproblems`: A dictionary mapping paths to Subproblem objects.
    - `cut_generators`: A dictionary mapping paths to CutGenerator objects.
    - `convex_approxs`: A dictionary mapping paths to ConvexApprox objects.
    - `model_info`: Information about the model.
"""
struct Tree
    root_node::Node
    nodes::Dict{Tuple{Vararg{Int}}, Node}
    subproblems::Dict{Tuple{Vararg{Int}}, Subproblem}
    cut_generators::Dict{Tuple{Vararg{Int}}, AbstractCutGenerator}
    convex_approxs::Dict{Tuple{Vararg{Int}}, ConvexApprox}
    model_info::ModelInfo
end

# ---------------------------------------------------
mutable struct RunStats
    iterations::Vector{Dict{Symbol, Any}} # Store per-iteration results
    timer_output::TimerOutput
    ext::Dict{Symbol, Any} # For any extra statistics to be stored
    function RunStats()
        return new([], timer_output, Dict{Symbol, Any}())
    end
end

struct Options
    log_file::Union{Nothing, String}
    iteration_limit::Union{Nothing, Int}
    time_limit::Union{Nothing, Float64}
    num_sample_paths::Int
    num_simulations::Int # number of simulations for statistical estimation after the forward pass.
    alpha::Float64 # alpha value for confidence interval
    cg_bound::Float64
    cg_time_limit::Float64
    cgsp_tol::Float64
    fixed_iter_tol::Float64
    regularization_param::Float64
    cg_acc_until::Int # Accelerated CG until this iteration
    cg_acc_level::Int # Accelerated CG level: (0: Benders 1: τ=0, 2: τ=0 and β[extended_space_vars]=0)
    obj_diff_tol::Float64 # Tolerance for objective value difference when adding cuts
    convergence_check_iterations::Int # Number of iterations to check for convergence
    convergence_check_tol::Float64 # Tolerance for convergence check
    cg_inner_iteration_limit::Int # Maximum number of inner iterations for CGMP-CGSP loop
end

"""
    Nested Benders Decomposition (NBD) Algorithm
    Contain the tree, options, and parameters for running the algorithm
"""
struct NBD
    tree::Tree
    runstats::RunStats
    options::Options
end
