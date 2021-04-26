from math import pi
from typing import Union

import biorbd
from casadi import MX, SX, if_else, vertcat, lt, gt
from bioptim import (
    OptimalControlProgram,
    DynamicsFcn,
    DynamicsFunctions,
    Dynamics,
    DynamicsList,
    Bounds,
    BoundsList,
    BiMapping,
    QAndQDotBounds,
    InitialGuess,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    Problem,
    NonLinearProgram,
    PlotType,
    Solution,
    Shooting
)

TL = 30
LD = 10
LR = 10
F = 0.5
R = 0.01
global_tau_max = [11, 0.1]

def custom_plot_callback(x: MX, q_to_plot: list) -> MX:
    """
    Create a used defined plot function with extra_parameters

    Parameters
    ----------
    x: MX
        The current states of the optimization
    q_to_plot: list
        The slice indices to plot

    Returns
    -------
    The value to plot
    """

    return x[q_to_plot, :]

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
    n_tau = nlp.shape["tau"]
    n_fatigable_ma = nlp.shape["tau"]
    n_fatigable_mr = nlp.shape["tau"]
    n_fatigable_mf = nlp.shape["tau"]

    nlp.mapping = {
        "q": BiMapping(range(nq), range(nq)),
        "qdot": BiMapping(range(nqdot), range(nqdot)),
        "tau": BiMapping(range(n_tau), range(n_tau)),
        "fatigable_tau_ma": BiMapping(range(n_fatigable_ma), range(n_fatigable_ma)),
        "fatigable_tau_mr": BiMapping(range(n_fatigable_mr), range(n_fatigable_mr)),
        "fatigable_tau_mf": BiMapping(range(n_fatigable_mf), range(n_fatigable_mf))
    }

    # n_fatigable = nlp.mapping["fatigable_tau"].to_first.len
    fatigable_ma = nlp.mapping["fatigable_tau_ma"].to_second.map(states[nq + nqdot:nq + nqdot + n_fatigable_ma + 1])
    fatigable_mr = nlp.mapping["fatigable_tau_mr"].to_second.map(
        states[nq + nqdot + n_fatigable_ma:nq + nqdot + n_fatigable_ma + n_fatigable_mr + 1])
    fatigable_mf = nlp.mapping["fatigable_tau_mr"].to_second.map(
        states[nq + nqdot + n_fatigable_ma + n_fatigable_mr:])

    fatigable_ma_dot = MX()
    fatigable_mr_dot = MX()
    fatigable_mf_dot = MX()
    effective_tau = MX()
    for i in range(nlp.shape["tau"]):

        ma = fatigable_ma[i, 0]
        mr = fatigable_mr[i, 0]
        mf = fatigable_mf[i, 0]
        TL = tau[i]/global_tau_max[i]  # Target load

        c = if_else(lt(ma, TL), if_else(gt(mr, TL-ma), LD * (TL - ma), LD * mr), LR * (TL - ma))

        madot = c - F * ma
        mrdot = -c + R * mf
        mfdot = F * ma - R * mf

        fatigable_ma_dot = vertcat(fatigable_ma_dot, madot)
        fatigable_mr_dot = vertcat(fatigable_mr_dot, mrdot)
        fatigable_mf_dot = vertcat(fatigable_mf_dot, mfdot)

        effective_tau = vertcat(effective_tau, ma*global_tau_max[i])

    qddot = nlp.model.ForwardDynamics(q, qdot, effective_tau).to_mx()

    return qdot, qddot, fatigable_ma_dot, fatigable_mr_dot, fatigable_mf_dot


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
    fatigable_ma_mx = MX()
    fatigable_mr_mx = MX()
    fatigable_mf_mx = MX()
    fatigable_ma = nlp.cx()
    fatigable_mr = nlp.cx()
    fatigable_mf = nlp.cx()

    for i in nlp.mapping["tau"].to_first.map_idx:
        fatigable_ma = vertcat(fatigable_ma, MX.sym("Fatigable_tau_ma_" + dof_names[i].to_string(), 1, 1))
        fatigable_mr = vertcat(fatigable_mr, MX.sym("Fatigable_tau_mr_" + dof_names[i].to_string(), 1, 1))
        fatigable_mf = vertcat(fatigable_mf, MX.sym("Fatigable_tau_mf_" + dof_names[i].to_string(), 1, 1))

    for i, _ in enumerate(nlp.mapping["tau"].to_second.map_idx):
        fatigable_ma_mx = vertcat(fatigable_ma_mx, nlp.cx.sym("Fatigable_tau_ma_mx_" + dof_names[i].to_string(), 1, 1))
        fatigable_mr_mx = vertcat(fatigable_mr_mx, nlp.cx.sym("Fatigable_tau_mr_mx_" + dof_names[i].to_string(), 1, 1))
        fatigable_mf_mx = vertcat(fatigable_mf_mx, nlp.cx.sym("Fatigable_tau_mf_mx_" + dof_names[i].to_string(), 1, 1))

    nlp.shape["fatigable_tau_ma"] = nlp.mapping["tau"].to_first.len
    nlp.shape["fatigable_tau_mr"] = nlp.mapping["tau"].to_first.len
    nlp.shape["fatigable_tau_mf"] = nlp.mapping["tau"].to_first.len

    nlp.fatigable_tau = {
        "fatigable_ma": MX(),
        "fatigable_mr": MX(),
        "fatigable_mf": MX()
    }

    nlp.fatigable_tau['fatigable_ma'] = fatigable_ma_mx
    nlp.fatigable_tau['fatigable_mr'] = fatigable_mr_mx
    nlp.fatigable_tau['fatigable_mf'] = fatigable_mf_mx
    nlp.x = vertcat(nlp.x, fatigable_ma)
    nlp.x = vertcat(nlp.x, fatigable_mr)
    nlp.x = vertcat(nlp.x, fatigable_mf)
    nlp.var_states["fatigable_tau_ma"] = nlp.shape["fatigable_tau_ma"]
    nlp.var_states["fatigable_tau_mr"] = nlp.shape["fatigable_tau_mr"]
    nlp.var_states["fatigable_tau_mf"] = nlp.shape["fatigable_tau_mf"]

    legend_fatigable_tau_ma = ["fatigable_tau_ma_" + nlp.model.nameDof()[idx].to_string() for idx in
                            nlp.mapping["tau"].to_first.map_idx]
    legend_fatigable_tau_mr = ["fatigable_tau_mr_" + nlp.model.nameDof()[idx].to_string() for idx in
                               nlp.mapping["tau"].to_first.map_idx]
    legend_fatigable_tau_mf = ["fatigable_tau_mf_" + nlp.model.nameDof()[idx].to_string() for idx in
                               nlp.mapping["tau"].to_first.map_idx]
    legend_fatigable_tau = legend_fatigable_tau_ma + legend_fatigable_tau_mr + legend_fatigable_tau_mf
    ocp.add_plot("My New Extra Plot",
                 lambda x, u, p: custom_plot_callback(x, [4, 5, 6, 7, 8, 9]),
                 plot_type=PlotType.PLOT,
                 legend=legend_fatigable_tau)

    nlp.nx = nlp.x.rows()

    Problem.configure_dynamics_function(ocp, nlp, custom_dynamic)



def prepare_ocp(biorbd_model_path: str,
                final_time: float,
                n_shooting: int,
                ode_solver: OdeSolver = OdeSolver.RK4(),
                use_sx: bool = False,
                ) -> OptimalControlProgram:
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
    dynamics = DynamicsList()
    dynamics.add(custom_configure, dynamic_function=custom_dynamic)

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
        ode_solver=ode_solver,
        use_sx=use_sx
    )


if __name__ == "__main__":
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    ocp = prepare_ocp(biorbd_model_path="pendulum.bioMod", final_time=3, n_shooting=100)

    # --- Test initial guess with single shooting --- #

    # n_tau = 2
    # n_q = 2
    # n_qdot = 2
    # tau_init = 0
    #
    # u_init = InitialGuess([tau_init] * n_tau)
    # x_init = InitialGuess([0] * (n_q + n_qdot))
    # x_init.concatenate(InitialGuess([0] * n_tau))  # ma
    # x_init.concatenate(InitialGuess([1] * n_tau))  # mr
    # x_init.concatenate(InitialGuess([0] * n_tau))  # mf
    #
    # sol_from_initial_guess = Solution(ocp, [x_init, u_init])
    # s = sol_from_initial_guess.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS)
    # print(f"Final position of q from single shooting of initial guess = {s.states['q'][:, -1]}")
    # s.animate()

    # --- Solve the ocp --- #
    sol = ocp.solve(show_online_optim=True)
    s_single = sol.integrate(shooting_type=Shooting.SINGLE_CONTINUOUS)
    sol.graphs(shooting_type=Shooting.SINGLE_CONTINUOUS)
    s_single.animate()

    # --- Show the results in a bioviz animation --- #
    sol.print()
    sol.graphs()
    sol.animate()