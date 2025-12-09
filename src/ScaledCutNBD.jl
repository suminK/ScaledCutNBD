module ScaledCutNBD

using JuMP
using TimerOutputs
using Random
using JSON
using ProgressMeter
using Logging
using InteractiveUtils
import Statistics.mean, Statistics.std, Statistics.quantile
import Distributions.TDist
import Printf: @sprintf
import Dates: now, format

export Tree, Subproblem, State, NBD, run

const timer_output = TimerOutput()
const VERSION = v"0.1.0"

# Setup debug logger
debug_mode = false
if debug_mode
    debug_logger = Logging.ConsoleLogger(stdout, Logging.Debug)
    Logging.global_logger(debug_logger)
end

include("types.jl")
include("utils/helper.jl")

include("JuMP.jl")
include("Subproblem.jl")
include("CutGenerator.jl")
include("AcceleratedCutGenerator.jl")
include("LagrangianCutGenerator.jl")
include("BendersCutGenerator.jl")
include("Tree.jl")

include("NBD.jl")
include("CutManagement.jl")

end