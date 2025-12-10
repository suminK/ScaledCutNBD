module Parser

export parse_args, validate_args

struct ParsingItem
    name::Symbol
    datatype::DataType
    required::Bool
    usage::String
end

const PARSING_ITEMS = Dict{Symbol,ParsingItem}(
    :method => ParsingItem(:method, Int, true, "1: SCNBD, 2: DEF"),
    :r => ParsingItem(:r, Int, true, "Depth parameter for SCNBD"),
    :T => ParsingItem(:T, Int, true, "Number of time periods"),
    :num_branch_each => ParsingItem(:num_branch_each, Int, true, "Number of branches at each time period"),
    :time_limit => ParsingItem(:time_limit, Float64, true, "Time limit in seconds"),
    :seed => ParsingItem(:seed, Int, true, "Random seed"),
    # problem-specific or optional parameters
    :iteration_limit => ParsingItem(:iteration_limit, Int, false, "iteration limit"),
    :cut_io_mode => ParsingItem(:cut_io_mode, String, false, "'r'=read, 'w'=write, 'rw'=read_and_write, else :none"),
    :cut_read_filename => ParsingItem(:cut_read_filename, String, false, "Filename for reading cut (.json)"),
    :cut_write_filename => ParsingItem(:cut_write_filename, String, false, "Filename for writing cuts (.json)"),
    :regularization_param => ParsingItem(:regularization_param, Float64, false, "regularization parameter for smoothing CGMP objective"),
    :num_item => ParsingItem(:num_item, Int, false, "[Lot-sizing] number of items"),
    :binarized_state_flag => ParsingItem(:binarized_state_flag, Int, false, "[Lot-sizing] Bool (1=true 0=false) whether to use binarized state variables"),
    :int_quantity_flag => ParsingItem(:int_quantity_flag, Int, false, "[Lot-sizing] Bool (1=true 0=false) whether to use integer quantities"),
    :decimal_factor => ParsingItem(:decimal_factor, Int, false, "decimal factor for binary expansion of state variables. E.g., 2 means to binarize to the second decimal places."),
    :cg_type => ParsingItem(:cg_type, Int, false, "1: CutGenerator, 2: AcceleratedCutGenerator, 3: LagrangianCutGenerator, 4: BendersCutGenerator"),
    :cg_acc_until => ParsingItem(:cg_acc_until, Int, false, "Iteration until which acceleration is used."),
    :cg_acc_level => ParsingItem(:cg_acc_level, Int, false, "Acceleration Level 0: Benders, 1: use NonScaledCuts, 2: use Non-scaled cuts and Non-extended space."),
    :def_presolve => ParsingItem(:def_presolve, Int, false, "[DEF] Bool (1=true 0=false) whether to presolve DEF model"),
    :cgsp_tol => ParsingItem(:cgsp_tol, Float64, false, "Tolerance for CGSP convergence"),
    :fixed_iter_tol => ParsingItem(:fixed_iter_tol, Float64, false, "Tolerance for fixed iteration convergence"),
    :cg_bound => ParsingItem(:cg_bound, Float64, false, "Bound for CG variables"),
    :print_timer => ParsingItem(:print_timer, Int, false, "[SCNBD] Bool (1=true 0=false) whether to print timer information after execution."),
)

function print_usage()
    println("Usage: julia main.jl [--key value] ...")
    println("Required keys: (DataType)")
    for (key, item) in PARSING_ITEMS
        if item.required
            println("  --$(key.name) ($(key.datatype)): $(key.usage)")
        end
    end
    println("Problems-specific / Optional keys: (DataType)")
    for (key, item) in PARSING_ITEMS
        if !item.required
            println("  --$(key.name) ($(key.datatype)): $(key.usage)")
        end
    end
end

"""
parse_args(defaults::Dict{Symbol,Any}=Dict())::Dict{Symbol,Any}
    Parses command-line arguments into a Dict.
    - additional_req_keys: Set of additional required keys.
    - defaults: Dict of default values for optionalarguments.
    Returns a Dict mapping argument names to their values.

    Note: Exit if no arguments are provided.
    Note: Input arguments have priority over defaults.
"""
function parse_args(
    ;
    additional_req_keys::Set{Symbol}=Set(),
    defaults::Dict{Symbol,Any}=Dict{Symbol,Any}(),
)::Dict{Symbol,Any}
    if length(ARGS) == 0
        print_usage()
        exit()
    end

    args = copy(ARGS)
    params = Dict{Symbol,Any}()

    i = 1
    while i <= length(args)
        arg = args[i]
        if startswith(arg, "--")
            key = Symbol(arg[3:end])
            if !haskey(PARSING_ITEMS, key)
                error("Unknown argument: $arg.")
            end
            if i == length(args)
                error("Missing value for argument $arg")
            end
            value_str = args[i+1]
            value = PARSING_ITEMS[key].datatype == String ? value_str :
                    PARSING_ITEMS[key].datatype == Char ? value_str[1] : tryparse(PARSING_ITEMS[key].datatype, value_str)
            if isnothing(value)
                error("Invalid value for argument $arg: expected $(PARSING_ITEMS[key].datatype), got '$value_str'")
            end
            params[key] = value
            i += 2
        else
            error("Unexpected argument format: $arg (expected --key value)")
        end
    end

    # Apply defaults only if the parameters are not given
    for (k, v) in defaults
        if !haskey(params, k)
            params[k] = v
        end
    end

    # Validate arguments
    for k in ([key for (key, val) in PARSING_ITEMS if val.required] ∪ additional_req_keys)
        if !haskey(params, k)
            error("Missing required argument: --$k")
        end
    end
    validate_args(params)

    # compute derived parameters
    params[:num_branch] = [params[:num_branch_each] for _ in 1:params[:T]]
    params[:num_branch][1] = 1
    if haskey(params, :cut_io_mode)
        params[:cut_io_mode] = params[:cut_io_mode] == "r" ? :r :
                               params[:cut_io_mode] == "w" ? :w :
                               params[:cut_io_mode] == "rw" ? :rw : :n
    end

    return params
end

function validate(condition::Bool, msg::String)
    if !condition
        error("Validation Error: $msg")
    end
end

"""
validate_args(param`s::Dict)
    Validates the parsed arguments in params Dict.
"""
function validate_args(params::Dict)
    validate(1 <= params[:method] <= 2, "method must be 1 (SCNBD), or 2 (DEF)")
    validate(params[:num_branch_each] >= 1, "num_branch_each must be ≥ 1")
    validate(params[:T] > 1, "T must be >= 2")
    if params[:method] == 1
        validate(1 <= params[:r] <= params[:T], "r must be between 1 and T")
    end
    validate(params[:time_limit] >= 0.0, "time_limit cannot be negative")
    validate(params[:seed] >= 0, "seed cannot be negative")
    #
    # problem-specific or optional parameters validation
    #
    if haskey(params, :cg_type)
        validate(1 <= params[:cg_type] <= 4, "cg_type must be 1 (CutGenerator), 2 (AcceleratedCutGenerator), 3 (LagrangianCutGenerator), or 4 (BendersCutGenerator)")
    end
    if haskey(params, :iteration_limit)
        validate(params[:iteration_limit] >= 1, "iteration_limit must be ≥ 1")
    end
    if haskey(params, :regularization_param)
        validate(params[:regularization_param] >= 0.0, "regularization_param must be ≥ 0.0")
    end
    if haskey(params, :cut_io_mode) && params[:cut_io_mode] in (:r, :rw)
        validate(haskey(params, :cut_read_filename), "cut_read_filename must be provided if cut_io_mode is 'r', or 'rw'")
    end
    if haskey(params, :cut_io_mode) && params[:cut_io_mode] in (:w, :rw)
        validate(haskey(params, :cut_write_filename), "cut_write_filename must be provided if cut_io_mode is 'w', or 'rw'")
    end
    if haskey(params, :num_item)
        validate(params[:num_item] >= 1, "num_item must be ≥ 1")
    end
    if haskey(params, :cg_type) && params[:cg_type] == 2
        if !(haskey(params, :cg_acc_until) && haskey(params, :cg_acc_level))
            error("cg_acc_until and cg_acc_level must be provided for AcceleratedCutGenerator (cg_type=2)")
        end
        validate(0 <= params[:cg_acc_until], "cg_acc_until must be ≥ 0")
        validate(params[:cg_acc_level] in (0, 1, 2), "cg_acc_level must be 0, 1, or 2")
    end
    if haskey(params, :binarized_state_flag)
        validate(params[:binarized_state_flag] in (0, 1), "binarized_state_flag must be 0 (false) or 1 (true)")
    end
    if haskey(params, :int_quantity_flag)
        validate(params[:int_quantity_flag] in (0, 1), "int_quantity_flag must be 0 (false) or 1 (true)")
    end
    if haskey(params, :decimal_factor)
        validate(params[:decimal_factor] >= 0, "decimal_factor must be >= 0")
    end
    if haskey(params, :def_presolve)
        validate(params[:def_presolve] in (0, 1), "def_presolve must be 0 (false) or 1 (true)")
    end
    if haskey(params, :cgsp_tol)
        validate(params[:cgsp_tol] > 0.0, "cgsp_tol must be > 0.0")
    end
    if haskey(params, :fixed_iter_tol)
        validate(params[:fixed_iter_tol] > 0.0, "fixed_iter_tol must be > 0.0")
    end
    if haskey(params, :cg_bound)
        validate(params[:cg_bound] > 0.0, "cg_bound must be > 0.0")
    end
    if haskey(params, :print_timer)
        validate(params[:print_timer] in (0, 1), "print_timer must be 0 (false) or 1 (true)")
    end
end

end