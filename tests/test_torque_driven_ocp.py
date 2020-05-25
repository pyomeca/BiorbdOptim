"""
Test for file IO
"""
import importlib.util
from pathlib import Path

import pytest
import numpy as np

from biorbd_optim import Data, OdeSolver, Constraint, Instant
from .utils import TestUtils

# Load align_markers
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/align_markers.py"
)
align_markers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(align_markers)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_align_markers(ode_solver):
    ocp = align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        number_shooting_points=30,
        final_time=2,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 19767.53312569522)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (186, 1))
    np.testing.assert_almost_equal(g, np.zeros((186, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((1.4516128810214546, 9.81, 2.2790322540381487)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-1.4516128810214546, 9.81, -2.2790322540381487)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


def test_align_markers_changing_constraints():
    ocp = align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod",
        number_shooting_points=30,
        final_time=2,
    )
    sol = ocp.solve()

    # Add a new constraint and reoptimize
    ocp.add_constraint(
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.MID, "first_marker_idx": 0, "second_marker_idx": 2,}
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 20370.211697123825)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((4.2641129, 9.81, 2.27903226)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((1.36088709, 9.81, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)

    # Replace constraints and reoptimize
    ocp.modify_constraint(
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.START, "first_marker_idx": 0, "second_marker_idx": 2,}, 0
    )
    ocp.modify_constraint(
        {"type": Constraint.ALIGN_MARKERS, "instant": Instant.MID, "first_marker_idx": 0, "second_marker_idx": 3,}, 2
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 31670.93770220887)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (189, 1))
    np.testing.assert_almost_equal(g, np.zeros((189, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((2, 0, 1.57)))
    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0)))
    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((-5.625, 21.06, 2.2790323)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((-5.625, 21.06, -2.27903226)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)


# Load multiphase_align_markers
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "multiphase_align_markers", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/multiphase_align_markers.py"
)
multiphase_align_markers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multiphase_align_markers)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK, OdeSolver.COLLOCATION])
def test_multiphase_align_markers(ode_solver):
    ocp = multiphase_align_markers.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube.bioMod", ode_solver=ode_solver
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 106084.82631762947)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (444, 1))
    np.testing.assert_almost_equal(g, np.zeros((444, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"], concatenate=False)
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[0][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[0][:, -1], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[1][:, 0], np.array((2, 0, 0)))
    np.testing.assert_almost_equal(q[1][:, -1], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[2][:, 0], np.array((1, 0, 0)))
    np.testing.assert_almost_equal(q[2][:, -1], np.array((2, 0, 1.57)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[0][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[0][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[1][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[1][:, -1], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[2][:, 0], np.array((0, 0, 0)))
    np.testing.assert_almost_equal(qdot[2][:, -1], np.array((0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[0][:, 0], np.array((1.42857142, 9.81, 0)))
    np.testing.assert_almost_equal(tau[0][:, -1], np.array((-1.42857144, 9.81, 0)))
    np.testing.assert_almost_equal(tau[1][:, 0], np.array((-0.2322581, 9.81, 0.0)))
    np.testing.assert_almost_equal(tau[1][:, -1], np.array((0.2322581, 9.81, -0.0)))
    np.testing.assert_almost_equal(tau[2][:, 0], np.array((0.35714285, 9.81, 0.56071428)))
    np.testing.assert_almost_equal(tau[2][:, -1], np.array((-0.35714285, 9.81, -0.56071428)))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)


# Load external_forces
PROJECT_FOLDER = Path(__file__).parent / ".."
spec = importlib.util.spec_from_file_location(
    "external_forces", str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/external_forces.py"
)
external_forces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(external_forces)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK])
def test_external_forces(ode_solver):
    ocp = external_forces.prepare_ocp(
        biorbd_model_path=str(PROJECT_FOLDER) + "/examples/torque_driven_ocp/cube_with_forces.bioMod",
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol["f"])
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 9875.88768746912)

    # Check constraints
    g = np.array(sol["g"])
    np.testing.assert_equal(g.shape, (246, 1))
    np.testing.assert_almost_equal(g, np.zeros((246, 1)))

    # Check some of the results
    states, controls = Data.get_data(ocp, sol["x"])
    q, qdot, tau = states["q"], states["q_dot"], controls["tau"]

    # initial and final position
    np.testing.assert_almost_equal(q[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(q[:, -1], np.array((0, 2, 0, 0)))

    # initial and final velocities
    np.testing.assert_almost_equal(qdot[:, 0], np.array((0, 0, 0, 0)))
    np.testing.assert_almost_equal(qdot[:, -1], np.array((0, 0, 0, 0)))

    # initial and final controls
    np.testing.assert_almost_equal(tau[:, 0], np.array((0, 9.71322593, 0, 0)))
    np.testing.assert_almost_equal(tau[:, 10], np.array((0, 7.71100122, 0, 0)))
    np.testing.assert_almost_equal(tau[:, 20], np.array((0, 5.70877651, 0, 0)))
    np.testing.assert_almost_equal(tau[:, -1], np.array((0, 3.90677425, 0, 0)))

    # save and load
    TestUtils.save_and_load(sol, ocp, True)
