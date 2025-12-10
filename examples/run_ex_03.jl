#=
    Example script to run SCNBD on the lot-sizing problem `examples/ex_03.jl`.
    We use the parser `examples/ex_03_parser.jl` to parse the command-line arguments.
    If you only need instance data for testing, refer to `examples/ex_03_data`.
=#

# input parameters
int_quantity_flag=0 # 0 (false) or 1 (true)
binarization_flag=0 # 0 (false) or 1 (true)
num_item=1 # ≥ 1
num_stage=3 # ≥ 2
method=1 # 1 (SCNBD) or 2 (DEF)
depth=1
num_branch_each=5
time_limit=600.0 # in seconds
seed=1

run(`julia --project="." examples/ex_03.jl \
      --method "$method" \
      --r "$depth" \
      --num_item "$num_item" \
      --num_branch_each "$num_branch_each" \
      --T "$num_stage" \
      --time_limit "$time_limit" \
      --seed "$seed" \
      --int_quantity_flag "$int_quantity_flag" \
      --binarized_state_flag "$binarization_flag"`
)