"""
    Write cuts to JSON file.
    - `nbd`: NBD object
    - `filename`: output filename
"""
function write_cuts(nbd::NBD, filename::String)
    convex_approxs = nbd.tree.convex_approxs
    cuts = Dict{Tuple{Vararg{Int}}, Vector{Dict{Symbol, Any}}}()
    # for each convex approx, get its cuts
    for (path, convex_approx) in convex_approxs
        cuts[path] = [Dict(:α => cut.α, :β => cut.β, :τ => cut.τ) for cut in convex_approx.cuts]
    end
    output = Dict{String, Any}()
    output["r"] = nbd.tree.model_info.r
    output["cuts"] = cuts
    open(filename, "w") do io
        JSON.print(io, output)
    end
end

"""
    Read cuts from JSON file and add them to the NBD object.
    - `nbd`: NBD object
    - `filename`: input filename
"""
function read_cuts(nbd::NBD, filename::String)
    nbd.runstats.ext[:cut_file_name] = filename

    # parse JSON file
    data = JSON.parsefile(filename)
    if data["r"] > nbd.tree.model_info.r
        error("The cut file was generated with r=$(data["r"]) which is greater than the current r=$(nbd.tree.model_info.r). \nCannot read cuts.")
    end

    # Read cuts
    println("Reading cuts from file: $filename")
    num_convex_approxs = length(data["cuts"])
    println("The number of convex approxs: $num_convex_approxs")

    # first, find all valid paths in the current NBD that correspond to the paths in the cut file
    path_mappings = Dict{Tuple{Vararg{Int}}, Vector{Tuple{Vararg{Int}}}}()
    for k in keys(data["cuts"])
        path = string_to_tuple(k)
        path_mappings[path] = Vector{Tuple{Vararg{Int}}}()
        for convex_approx_path in keys(nbd.tree.convex_approxs)
            if length(convex_approx_path) != length(path)
                continue
            end
            # Check if `convex_approx_path` is a valid extension of `path`
            is_valid = true
            for (i, j) in enumerate(path)
                if j != 0 && convex_approx_path[i] != j
                    is_valid = false
                    break
                end
            end
            if is_valid
                push!(path_mappings[path], convex_approx_path)
            end
        end
    end

    total_valid_paths = sum(length(v) for v in values(path_mappings))
    println("Total valid paths found: $total_valid_paths")
    total_actions = sum(length(data["cuts"][string(k)]) * length(v) for (k, v) in path_mappings)
    println("The total number of cut additions to be performed: $total_actions")

    progress = Progress(total_actions, 1)
    for (target_path, valid_paths) in path_mappings
        for path in valid_paths
            convex_approx = nbd.tree.convex_approxs[path]
            for cut_data in data["cuts"][string(target_path)]
                ProgressMeter.next!(progress; step=1)
                add_cut_to_convex_approx(convex_approx, cut_data)
            end
        end
    end

    ProgressMeter.finish!(progress)
    println("Finished reading cuts.\n")
end

function add_cut_to_convex_approx(convex_approx::ConvexApprox, cut_data::Dict{String, Any})
    α_val = cut_data["α"]
    τ_val = cut_data["τ"]
    intercept = α_val / (1 + τ_val)
    β_val = Dict{Symbol, Float64}()
    slope = Dict{Symbol, Float64}()
    for (β_k, β_v) in cut_data["β"]
        β_val[Symbol(β_k)] = β_v
        slope[Symbol(β_k)] = - β_v / (1 + τ_val)
    end
    # Add cut to convex_approx
    cut = ScaledCut(α_val, β_val, τ_val)
    push!(convex_approx.cuts, cut)
    for model_hook in convex_approx.model_hooks
        model_hook.add_cut_func(slope, intercept)
    end
end

function string_to_tuple(s::String)::Tuple{Vararg{Int}}
    s = strip(s, ['(', ')', ' ', ','])
    isempty(s) && return ()
    return Tuple(parse.(Int, split(s, ',')))
end
