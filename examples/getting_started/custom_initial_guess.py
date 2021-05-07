"""
This example is a trivial box that must superimpose one of its corner to a marker at the beginning of the movement
and superimpose the same corner to a different marker at the end.
It is designed to investigate the different way to define the initial guesses at each node sent to the solver

All the types of interpolation are shown:
InterpolationType.CONSTANT: All the values are the same at each node
InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT: Same as constant, but have the first
    and last nodes different. This is particularly useful when you want to fix the initial and
    final position and leave the rest of the movement free.
InterpolationType.LINEAR: The values are linearly interpolated between the first and last nodes.
InterpolationType.EACH_FRAME: Each node values are specified
InterpolationType.SPLINE: The values are interpolated from the first to last node using a cubic spline
InterpolationType.CUSTOM: Provide a user-defined interpolation function
"""

import numpy as np
import biorbd
from bioptim import (
    Node,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    Objective,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    Bounds,
    QAndQDotBounds,
    InitialGuess,
    InterpolationType,
    OdeSolver,
)


def custom_init_func(current_shooting_point: int, my_values: np.ndarray, n_shooting: int) -> np.ndarray:
    """
    The custom function for the x and u initial guesses (this particular one mimics linear interpolation)

    Parameters
    ----------
    current_shooting_point: int
        The current point to return the value, it is defined between [0; n_shooting] for the states
        and [0; n_shooting[ for the controls
    my_values: np.ndarray
        The values provided by the user
    n_shooting: int
        The number of shooting point

    Returns
    -------
    The vector value of the initial guess at current_shooting_point
    """

    # Linear interpolation created with custom function
    return my_values[:, 0] + (my_values[:, -1] - my_values[:, 0]) * current_shooting_point / n_shooting


def prepare_ocp(
    biorbd_model_path: str,
    n_shooting: int,
    final_time: float,
    initial_guess: InterpolationType = InterpolationType.CONSTANT,
    ode_solver=OdeSolver.RK4(),
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    n_shooting: int
        The number of shooting points
    final_time: float
        The time at the final node
    initial_guess: InterpolationType
        The type of interpolation to use for the initial guesses
    ode_solver: OdeSolver
        The type of ode solver used

    Returns
    -------
    The ocp ready to be solved
    """

    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    nq = biorbd_model.nbQ()
    nqdot = biorbd_model.nbQdot()
    ntau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_TORQUE, weight=100)

    # Dynamics
    dynamics = Dynamics(DynamicsFcn.TORQUE_DRIVEN)

    # Constraints
    constraints = ConstraintList()
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.START, first_marker_idx=0, second_marker_idx=1)
    constraints.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker_idx=0, second_marker_idx=2)

    # Path constraint and control path constraints
    x_bounds = QAndQDotBounds(biorbd_model)
    x_bounds[1:6, [0, -1]] = 0
    x_bounds[2, -1] = 1.57
    u_bounds = Bounds([tau_min] * ntau, [tau_max] * ntau)

    # Initial guesses
    t = None
    extra_params_x = {}
    extra_params_u = {}
    if initial_guess == InterpolationType.CONSTANT:
        x = [0] * (nq + nqdot)
        u = [tau_init] * ntau
    elif initial_guess == InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [1.5, 0.0, 0.785, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [0, 9.81, 0], [-1.45, 9.81, -2.28]]).T
    elif initial_guess == InterpolationType.LINEAR:
        x = np.array([[1.0, 0.0, 0.0, 0, 0, 0], [2.0, 0.0, 1.57, 0, 0, 0]]).T
        u = np.array([[1.45, 9.81, 2.28], [-1.45, 9.81, -2.28]]).T
    elif initial_guess == InterpolationType.EACH_FRAME:
        x = np.random.random((nq + nqdot, n_shooting + 1))
        u = np.random.random((ntau, n_shooting))
    elif initial_guess == InterpolationType.SPLINE:
        # Bound spline assume the first and last point are 0 and final respectively
        t = np.hstack((0, np.sort(np.random.random((3,)) * final_time), final_time))
        x = np.random.random((nq + nqdot, 5))
        u = np.random.random((ntau, 5))
    elif initial_guess == InterpolationType.CUSTOM:
        # The custom function refers to the one at the beginning of the file. It emulates a Linear interpolation
        x = custom_init_func
        u = custom_init_func
        extra_params_x = {"my_values": np.random.random((nq + nqdot, 2)), "n_shooting": n_shooting}
        extra_params_u = {"my_values": np.random.random((ntau, 2)), "n_shooting": n_shooting}
    else:
        raise RuntimeError("Initial guess not implemented yet")
    x_init = InitialGuess(x, t=t, interpolation=initial_guess, **extra_params_x)

    u_init = InitialGuess(u, t=t, interpolation=initial_guess, **extra_params_u)
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
    Solve the program for all the InterpolationType available
    """

    for initial_guess in InterpolationType:
        print(f"Solving problem using {initial_guess} initial guess")
        ocp = prepare_ocp("cube.bioMod", n_shooting=30, final_time=2, initial_guess=initial_guess)

        # --- Print ocp structure --- #
        ocp.print_ocp_structure()

        sol = ocp.solve()
        print("\n")

    # Print the last solution
    sol.animate()
