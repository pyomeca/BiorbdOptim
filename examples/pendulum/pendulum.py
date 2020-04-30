import biorbd

from biorbd_optim import OptimalControlProgram, OdeSolver
from biorbd_optim.plot import PlotOcp
from biorbd_optim.problem_type import ProblemType
from biorbd_optim.objective_functions import ObjectiveFunction
from biorbd_optim.constraints import Constraint
from biorbd_optim.path_conditions import Bounds, QAndQDotBounds, InitialConditions


def prepare_ocp(biorbd_model_path="eocar.bioMod"):
    # --- Options --- #
    # Model path
    biorbd_model = biorbd.Model(biorbd_model_path)

    # Problem parameters
    number_shooting_points = 30
    final_time = 2
    torque_min, torque_max, torque_init = -100, 100, 0

    # Add objective functions
    objective_functions = ((ObjectiveFunction.minimize_torque, 100),)


    # Dynamics
    variable_type = ProblemType.torque_driven

    # Constraints
    constraints = ()

    # Path constraint
    X_bounds = QAndQDotBounds(biorbd_model)

    for i in range(biorbd_model.nbQ() + biorbd_model.nbQdot()):
        X_bounds.first_node_min[i] = 0
        X_bounds.last_node_min[i] = 0
        X_bounds.first_node_max[i] = 0
        X_bounds.last_node_max[i] = 0

    #Q_rotation
    X_bounds.last_node_min[1] = 3.14
    X_bounds.last_node_max[1] = 3.14

    #Qdot_translation
    X_bounds.last_node_min[2] = 1
    X_bounds.last_node_max[2] = 2


    # for i in range(1, 6):
    #     X_bounds.first_node_min[i] = 0
    #     X_bounds.last_node_min[i] = 0
    #     X_bounds.first_node_max[i] = 0
    #     X_bounds.last_node_max[i] = 0
    # X_bounds.last_node_min[2] = 1.57
    # X_bounds.last_node_max[2] = 1.57

    # Initial guess
    X_init = InitialConditions([2] * (biorbd_model.nbQ() + biorbd_model.nbQdot()))

    # Define control path constraint
    U_bounds = Bounds(
        [torque_min] * biorbd_model.nbGeneralizedTorque(),
        [torque_max] * biorbd_model.nbGeneralizedTorque(),
    )
    U_init = InitialConditions([torque_init] * biorbd_model.nbGeneralizedTorque())

    # ------------- #

    return OptimalControlProgram(
        biorbd_model,
        variable_type,
        number_shooting_points,
        final_time,
        objective_functions,
        X_init,
        U_init,
        X_bounds,
        U_bounds,
        constraints,
        ode_solver=OdeSolver.COLLOCATION,
        show_online_optim=True,
    )


if __name__ == "__main__":
    ocp = prepare_ocp()

    # --- Solve the program --- #
    sol = ocp.solve()

    #--- Plot ---#
    x, _, _ = ProblemType.get_data_from_V(ocp, sol["x"])
    x = ocp.nlp[0]["dof_mapping"].expand(x)

    try:
        from BiorbdViz import BiorbdViz

        b = BiorbdViz(loaded_model=ocp.nlp[0]["model"])
        b.load_movement(x.T)
        b.exec()
    except ModuleNotFoundError:
        print("Install BiorbdViz if you want to have a live view of the optimization")


    # plt_ocp = PlotOcp(ocp)
    # plt_ocp.update_data(sol["x"])
    # plt_ocp.show()


