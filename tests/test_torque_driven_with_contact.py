"""
Test for file IO.
It tests the results of an optimal control problem with torque_driven_with_contact problem type regarding the proper functioning of :
- the maximize/minimize_predicted_height_CoM objective
- the contact_forces_inequality constraint
- the non_slipping constraint
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import Data, OdeSolver

# Load symmetry
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "maximize_predicted_height_CoM",
    str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/maximize_predicted_height_CoM.py",
)
maximize_predicted_height_CoM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maximize_predicted_height_CoM)

spec = importlib.util.spec_from_file_location(
    "contact_forces_inequality_constraint",
    str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/contact_forces_inequality_constraint.py",
)
contact_forces_inequality_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contact_forces_inequality_constraint)

spec = importlib.util.spec_from_file_location(
    "non_slipping_constraint", str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/non_slipping_constraint.py",
)
non_slipping_constraint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(non_slipping_constraint)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_maximize_predicted_height_CoM(ode_solver):
    ocp = maximize_predicted_height_CoM.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.5,
        number_shooting_points=20,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.2370021957829777)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (160, 1))
    # np.testing.assert_almost_equal(g, np.zeros((160, 1)))

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-0.2191102, 0.0607841, -0.0412168, 0.0412168)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-0.3689301, 0.0152147, 0.7384873, -0.7384873)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-27.2604552)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-0.2636804)))


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_contact_forces_inequality_GREATER_THAN_constraint(ode_solver):
    ocp = contact_forces_inequality_constraint_GREATER_THAN.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        direction="GREATER_THAN"
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.1452562070438664)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (100, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)), decimal=6)
    np.testing.assert_array_less(-g[80:], 0)
    expected_pos_g = np.array(
        [
            [4.14326709e01],
            [5.89469050e01],
            [6.31668706e01],
            [6.62340130e01],
            [6.81979220e01],
            [6.85259469e01],
            [6.66231189e01],
            [6.18915213e01],
            [5.39938053e01],
            [4.34594467e01],
            [2.057260303e02],
            [9.16286233e01],
            [6.65898730e01],
            [5.81849733e01],
            [5.50995044e01],
            [5.44123368e01],
            [5.48895651e01],
            [5.55906536e01],
            [5.52487885e01],
            [5.19858748e01],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_pos_g)

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-3.4081740e-01, 1.3415556e-01, -3.9566794e-06, 3.9566794e-06)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.0922416e00, 8.2778459e-06, 4.1844833e00, -4.1844833e00)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-70.1205906)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-17.5866089)))

@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_contact_forces_inequality_LESSER_THAN_constraint(ode_solver):
    ocp = contact_forces_inequality_constraint_LESSER_THAN.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.3,
        number_shooting_points=10,
        direction="LESSER_THAN"
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.1452562070438664)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (100, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)), decimal=6)
    np.testing.assert_array_less(-g[80:], 0)
    expected_pos_g = np.array(
        [
            [4.14326709e01],
            [5.89469050e01],
            [6.31668706e01],
            [6.62340130e01],
            [6.81979220e01],
            [6.85259469e01],
            [6.66231189e01],
            [6.18915213e01],
            [5.39938053e01],
            [4.34594467e01],
            [2.057260303e02],
            [9.16286233e01],
            [6.65898730e01],
            [5.81849733e01],
            [5.50995044e01],
            [5.44123368e01],
            [5.48895651e01],
            [5.55906536e01],
            [5.52487885e01],
            [5.19858748e01],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_pos_g)

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.75, 0.75)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-3.4081740e-01, 1.3415556e-01, -3.9566794e-06, 3.9566794e-06)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-2.0922416e00, 8.2778459e-06, 4.1844833e00, -4.1844833e00)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-70.1205906)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-17.5866089)))

@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_non_slipping_constraint(ode_solver):
    ocp = non_slipping_constraint.prepare_ocp(
        model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_with_contact/2segments_4dof_2contacts.bioMod",
        phase_time=0.6,
        number_shooting_points=10,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.21820306105763151)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (110, 1))
    np.testing.assert_almost_equal(g[:80], np.zeros((80, 1)))
    np.testing.assert_array_less(-g[80:], 0)
    expected_pos_g = np.array(
        [
            [8.06654187e01],
            [8.42778524e01],
            [8.81615178e01],
            [9.17459644e01],
            [9.50899567e01],
            [9.83262204e01],
            [1.0156872164e02],
            [1.0488130122e02],
            [1.0821887319e02],
            [1.1131692919e02],
            [6.17068815e01],
            [5.34409324e01],
            [4.59011530e01],
            [4.06703203e01],
            [3.67591700e01],
            [3.32903343e01],
            [2.97727890e01],
            [2.59507626e01],
            [2.17350095e01],
            [1.72057582e01],
            [1.91986775e01],
            [2.40008630e01],
            [2.62549528e01],
            [2.68785755e01],
            [2.65984026e01],
            [2.57205116e01],
            [2.43993580e01],
            [2.27149568e01],
            [2.06672767e01],
            [1.81470345e01],
        ]
    )
    np.testing.assert_almost_equal(g[80:], expected_pos_g)

    # Check some of the results
    states, controls = Data.get_data_from_V(ocp, sol["x"])
    q = states["q"].to_matrix()
    qdot = states["q_dot"].to_matrix()
    tau = controls["tau"].to_matrix()

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0.0, 0.0, -0.5, 0.5)))
    np.testing.assert_almost_equal(q[:, -1], np.array((-2.3970871e-01, 6.1208704e-02, -8.1394939e-06, 8.1394939e-06)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((-7.4541514e-01, 5.9274858e-06, 1.4908303e00, -1.4908303e00)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-19.7138885)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-4.2970368)))