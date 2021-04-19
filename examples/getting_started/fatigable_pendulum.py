from math import pi
from typing import Union

import biorbd
from casadi import MX, SX, if_else, vertcat, lt, gt
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsFunctions,
    Dynamics,
    Bounds,
    BoundsList,
    BiMapping,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    Problem,
    NonLinearProgram,
    #  PlotType
)

TL = 30
LD = 10
LR = 10
F = 0.1
R = 0.02
global_tau_max = [100, 0]

def custom_dynamic(
    states: Union[MX, SX], controls: Union[MX, SX], parameters: Union[MX, SX], nlp: NonLinearProgram
) -> tuple:
    """
    The custom dynamics function that provides the derivative of the states: dxdt = f(x, u, p)

    Parameters
    ----------
    states: Union[MX, SX]
        The state of the system
    controls: Union[MX, SX]
        The controls of the system
    parameters: Union[MX, SX]
        The parameters acting on the system
    nlp: NonLinearProgram
        A reference to the phase

    Returns
    -------
    The derivative of the states in the tuple[Union[MX, SX]] format
    """

    DynamicsFunctions.apply_parameters(parameters, nlp)
    q, qdot, tau = DynamicsFunctions.dispatch_q_qdot_tau_data(states, controls, nlp)

    nq = nlp.shape["q"]
    nqdot = nlp.shape["qdot"]

    n_fatigable = nlp.mapping["fatigable_tau"].to_first.len
    fatigable = nlp.mapping["fatigable_tau"].to_second.map(states[nq + nqdot:])

    fatigable_dot = MX()
    effective_tau = MX()
    for i in range(nlp.shape["tau"]):

        ma = fatigable[0 + 3*i, 0]
        mr = fatigable[1 + 3*i, 0]
        mf = fatigable[2 + 3 * i, 0]
        TL = tau[i]/global_tau_max[i]

        c= if_else (lt(ma, TL), if_else(gt(mr, TL-ma), LD * (TL - ma), LD * mr), LR * (TL - ma))

        madot = c - F * ma
        mrdot = -c + R * mf
        mfdot = F * ma - R * mf

        fatigable_dot = vertcat(fatigable_dot, madot)
        fatigable_dot = vertcat(fatigable_dot, mrdot)
        fatigable_dot = vertcat(fatigable_dot, mfdot)

        effective_tau = vertcat(effective_tau, ma*global_tau_max[i])

    qddot = nlp.model.ForwardDynamics(q, qdot, effective_tau).to_mx()

    return qdot, qddot, fatigable_dot


def custom_configure(ocp: OptimalControlProgram, nlp: NonLinearProgram):
    """
    Tell the program which variables are states and controls.
    The user is expected to use the Problem.configure_xxx functions.

    Parameters
    ----------
    ocp: OptimalControlProgram
        A reference to the ocp
    nlp: NonLinearProgram
        A reference to the phase
    """

    Problem.configure_q_qdot(nlp, as_states=True, as_controls=False)
    Problem.configure_tau(nlp, as_states=False, as_controls=True)

    if nlp.mapping["tau"] is None:
        nlp.mapping["tau"] = BiMapping(range(nlp.model.nbGeneralizedTorque()), range(nlp.model.nbGeneralizedTorque()))

    dof_names = nlp.model.nameDof()
    fatigable_mx = MX()
    fatigable = nlp.cx()

    for i in nlp.mapping["tau"].to_first.map_idx:
        fatigable = vertcat(fatigable, nlp.cx.sym("Fatigable_tau_" + dof_names[i].to_string(), 3, 1))
    for i, _ in enumerate(nlp.mapping["tau"].to_second.map_idx):
        fatigable_mx = vertcat(fatigable_mx, MX.sym("Fatigable_tau_mx_" + dof_names[i].to_string(), 3, 1))

    nlp.shape["fatigable_tau"] = nlp.mapping["tau"].to_first.len*3

    legend_fatigable_tau = ["fatigable_tau_" + nlp.model.nameDof()[idx].to_string() for idx in nlp.mapping["tau"].to_first.map_idx]

    nlp.fatigable = fatigable_mx
    nlp.x = vertcat(nlp.x, fatigable)
    nlp.var_states["fatigable_tau"] = nlp.shape["fatigable_tau"]
    # q_bounds = nlp.x_bounds[nlp.shape["q"]+nlp.shape["qdot"]: nlp.shape["q"]+nlp.shape["qdot"]+nlp.shape["fatigable_tau"]]

    # nlp.plot["fatigable_tau"] = CustomPlot(
    #     lambda x, u, p: x[nlp.shape["q"]+nlp.shape["qdot"]: nlp.shape["q"]+nlp.shape["qdot"]+nlp.shape["fatigable_tau"]],
    #     plot_type=PlotType.INTEGRATED,
    #     legend=legend_q,
    #     bounds=q_bounds,
    # )

    nlp.nx = nlp.x.rows()

    Problem.configure_dynamics_function(ocp, nlp, custom_dynamic)



def prepare_ocp(biorbd_model_path: str, final_time: float, n_shooting: int) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: float
        The time in second required to perform the task
    n_shooting: int
        The number of shooting points to define int the direct multiple shooting program

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """

    biorbd_model = biorbd.Model(biorbd_model_path)

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()

    trans_q_bounds_min = [0, -1, 0]
    rot_q_bounds_min = [0, -2*pi, pi]
    trans_qdot_bounds_min = [0, -31.415925654, 0]
    rot_qdot_bounds_min = [0, -31.415925654, 0]
    ma_bounds_min = [0, 0, 0]
    mr_bounds_min = [1, 0, 0]
    mf_bounds_min = [0, 0, 0]
    min_bounds = [trans_q_bounds_min,
                  rot_q_bounds_min,
                  trans_qdot_bounds_min,
                  rot_qdot_bounds_min,
                  ma_bounds_min,
                  ma_bounds_min,
                  mr_bounds_min,
                  mr_bounds_min,
                  mf_bounds_min,
                  mf_bounds_min]

    trans_q_bounds_max = [0, 5, 0]
    rot_q_bounds_max = [0, 2 * pi, pi]
    trans_qdot_bounds_max = [0, 31.415925654, 0]
    rot_qdot_bounds_max = [0, 31.415925654, 0]
    ma_bounds_max = [0, 1, 1]
    mr_bounds_max = [1, 1, 1]
    mf_bounds_max = [0, 1, 1]
    max_bounds = [trans_q_bounds_max,
                  rot_q_bounds_max,
                  trans_qdot_bounds_max,
                  rot_qdot_bounds_max,
                  ma_bounds_max,
                  ma_bounds_max,
                  mr_bounds_max,
                  mr_bounds_max,
                  mf_bounds_max,
                  mf_bounds_max]

    x_bounds.add(min_bounds, max_bounds)

    # Define ma, mr, mf as states with initial_state = (ma, mr, mf) = (0, 1, 0) and final_state = (0, 0, 1)
    # 0 ≤ ma, mr, mf ≤ 1

    # x_bounds.min[[4, 5, 6], :] = 0
    # x_bounds.max[[4, 5, 6], :] = 1
    #
    # initial_state = (0, 1, 0)
    # x_bounds[4, 0] = initial_state[0]  # ma
    # x_bounds[5, 0] = initial_state[1]  # mr
    # x_bounds[6, 0] = initial_state[2]  # mf
    #
    # final_state = (0, 0, 1)
    # x_bounds[4, -1] = final_state[0]  # ma
    # x_bounds[5, -1] = final_state[1]  # mr
    # x_bounds[6, -1] = final_state[2]  # mf

    # Initial guess
    n_tau = biorbd_model.nbGeneralizedTorque()

    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    x_init = InitialGuess([0] * (n_q + n_qdot))
    x_init.concatenate(InitialGuess([0] * n_tau))  # ma
    x_init.concatenate(InitialGuess([1] * n_tau))  # mr
    x_init.concatenate(InitialGuess([0] * n_tau))  # mf

    # Define control path constraint
    tau_min, tau_max, tau_init = -100, 100, 0
    u_bounds = Bounds([tau_min] * n_tau, [tau_max] * n_tau)
    u_bounds[n_tau - 1, :] = 0

    u_init = InitialGuess([tau_init] * n_tau)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


if __name__ == "__main__":
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100)

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show the results in a bioviz animation --- #
    sol.print()
    sol.animate()