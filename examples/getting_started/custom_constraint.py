"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to show how one can define its own custom constraints function if the provided ones are not
sufficient.

More specifically this example reproduces the behavior of the SUPERIMPOSE_MARKERS constraint.
"""

import biorbd
from casadi import MX
from bioptim import (
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    PenaltyNode,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    OdeSolver,
)


def custom_func_track_markers(pn: PenaltyNode, first_marker_idx: int, second_marker_idx: int) -> MX:
    """
    The used-defined constraint function (This particular one mimics the ConstraintFcn.SUPERIMPOSE_MARKERS)
    Except for the last two

    Parameters
    ----------
    pn: PenaltyNode
        The penalty node elements
    first_marker_idx: int
        The index of the first marker in the bioMod
    second_marker_idx: int
        The index of the second marker in the bioMod

    Returns
    -------
    The value that should be constrained in the MX format
    """

    nq = pn.nlp.shape["q"]
    markers = pn.nlp.add_casadi_func("markers", pn.nlp.model.markers, pn.nlp.q)
    q = pn.x[:nq]
    first_marker = markers(q)[:, first_marker_idx]
    second_marker = markers(q)[:, second_marker_idx]
    return first_marker - second_marker


def prepare_ocp(biorbd_model_path: str, ode_solver: OdeSolver = OdeSolver.RK4()) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    n_shooting = 30
    final_time = 2
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(custom_func_track_markers, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(custom_func_track_markers, node=Node.END, first_marker_idx=0, second_marker_idx=2)

    # Path constraint
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57

    # Initial guess
    x_init = InitialGuess([0] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    u_bounds = Bounds([tau_min] * biorbd_model.nbGeneralizedTorque(), [tau_max] * biorbd_model.nbGeneralizedTorque())

    u_init = InitialGuess([tau_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        n_shooting,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions,
        constraints,
        ode_solver=ode_solver,
    )


if __name__ == "__main__":
    """
    Solve and animate the solution
    """

    model_path = "cube.bioMod"
    ocp = prepare_ocp(biorbd_model_path=model_path)

    # --- Solve the program --- #
    sol = ocp.solve(show_online_optim=True)

    # --- Show results --- #
    sol.animate()
