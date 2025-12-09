"""
    Extend @variable to State Variables.
"""
function JuMP.build_variable(
    error_fn::Function,
    var_info::JuMP.VariableInfo,
    ::Type{State};
    initial_value = NaN,
    kwargs...,
)
    if initial_value === NaN
        error_fn("You must set the `initial_value` of the state variables.")
    end
    return StateInfo(
        JuMP.VariableInfo(false, NaN, false, NaN, false, NaN, false, NaN, false, false),
        var_info,
        initial_value,
        kwargs,
    )
end

function JuMP.add_variable(
    model::JuMP.Model,
    state_info::StateInfo,
    name::String,
)
    t = model.ext[:stage]
    aug_name = "$(name)_$(t)"
    state = State(
        JuMP.add_variable(
            model,
            JuMP.ScalarVariable(state_info.in),
            aug_name * "_in",
        ),
        JuMP.add_variable(
            model,
            JuMP.ScalarVariable(state_info.out),
            aug_name * "_out",
        ),
        Symbol(name),
    )
    # Fix initial state value for the root node.
    if t == 1
        fix(state.in, state_info.initial_value)
    end
    # save the initial state values for later use.
    if !haskey(model.ext, :initial_state_values)
        model.ext[:initial_state_values] = Dict{Symbol, Float64}()
    end
    model.ext[:initial_state_values][Symbol(name)] = state_info.initial_value
    # add state variable reference to the subproblem object.
    add_var_ref!(model, Symbol(aug_name), state, :state)
    return state
end

function JuMP.value(state::State{JuMP.VariableRef})
    return State(JuMP.value(state.in), JuMP.value(state.out), state.original_name)
end

Broadcast.broadcastable(state::State{JuMP.VariableRef}) = Ref(state)
