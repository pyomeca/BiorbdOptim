import numpy as np
import biorbd

from biorbd_optim import OptimalControlProgram
from biorbd_optim.constraints import Constraint
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions
from biorbd_optim.plot import ShowResult


def prepare_ocp(biorbd_model_path="HandSpinner.bioMod", show_online_optim=False):
    end_crank_idx = 0
    hand_marker_idx = 18

    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)
    torque_min, torque_max, torque_init = -100, 100, 0
    muscle_min, muscle_max, muscle_init = 0, 1, 0.5

    # Problem parameters
    number_shooting_points = 30
    final_time = 1.5

    # Add objective functions
    objective_functions = (
        {"type": ObjectiveFunction.minimize_markers_displacement, "weight": 1, "markers_idx": hand_marker_idx},
        {"type": ObjectiveFunction.minimize_muscle, "weight": 1,},
        {"type": ObjectiveFunction.minimize_torque, "weight": 1,},
    )

    # Dynamics
    problem_type = ProblemType.muscles_and_torque_driven

    # Constraints
    constraints = (
        {
            "type": Constraint.Type.MARKERS_TO_MATCH,
            "first_marker": hand_marker_idx,
            "second_marker": end_crank_idx,
            "instant": Constraint.Instant.ALL,
        },
        {
            "type": Constraint.Type.TRACK_Q,
            "instant": Constraint.Instant.ALL,
            "states_idx": 0,
            "data_to_track": np.linspace(0, 2 * np.pi, number_shooting_points + 1),
        },
    )

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    # Initial guess
    X_init = InitialConditions([0, -0.9, 1.7, 0.9, 2.0, -1.3] + [0] * biorbd_model.nbQdot())

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque() + [muscle_min] * biorbd_model.nbMuscleTotal(),
        [torque_max] * biorbd_model.nbGeneralizedTorque() + [muscle_max] * biorbd_model.nbMuscleTotal(),
    )

    U_init = InitialConditions(
        [torque_init] * biorbd_model.nbGeneralizedTorque() + [muscle_init] * biorbd_model.nbMuscleTotal()
    )
    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        problem_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        is_cyclic_objective=True,
        show_online_optim=show_online_optim,
    )


if __name__ == "__main__":
    ocp = prepare_ocp(show_online_optim=True)

    # --- Solve the program --- #
    sol = ocp.solve()

    # --- Show results --- #
    result = ShowResult(ocp, sol)
    result.animate(show_meshes=False)
    result.graphs()
