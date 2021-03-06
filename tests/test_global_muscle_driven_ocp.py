"""
Test for file IO
"""
import pytest
import numpy as np
from bioptim import OdeSolver

from .utils import TestUtils


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4, OdeSolver.RK8, OdeSolver.IRK])
def test_muscle_driven_ocp(ode_solver):
    bioptim_folder = TestUtils.bioptim_folder()
    static_arm = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/static_arm.py")
    ode_solver = ode_solver()

    ocp = static_arm.prepare_ocp(
        bioptim_folder + "/examples/muscle_driven_ocp/arm26.bioMod",
        final_time=2,
        n_shooting=10,
        weight=1,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check constraints
    g = np.array(sol.constraints)
    np.testing.assert_equal(g.shape, (40, 1))
    np.testing.assert_almost_equal(g, np.zeros((40, 1)), decimal=6)

    # Check some of the results
    q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14351611580879933)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.94511299, 3.07048865]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.41149114, -0.55863385]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00147561, 0.00520749]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00027953, 0.00069257]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.29029533e-06, 1.64976642e-01, 1.00004898e-01, 4.01974257e-06, 4.13014984e-06, 1.03945583e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25940361e-03, 3.21754460e-05, 3.12984790e-05, 2.00725054e-03, 1.99993619e-03, 1.81725854e-03]),
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14350914060136277)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.94510844, 3.07048231]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.41151235, -0.55866253]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.00147777, 0.00520795]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.00027953, 0.00069258]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.28863414e-06, 1.65011897e-01, 1.00017224e-01, 4.01934660e-06, 4.12974244e-06, 1.03954780e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25990460e-03, 3.21893307e-05, 3.13077447e-05, 2.01209936e-03, 2.00481801e-03, 1.82353344e-03]),
        )

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14350464848810182)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([-0.9451058, 3.0704789]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.4115254, -0.5586797]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([0.0014793, 0.0052082]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-0.0002795, 0.0006926]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.2869218e-06, 1.6503522e-01, 1.0002514e-01, 4.0190181e-06, 4.1294041e-06, 1.0396051e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.2599283e-03, 3.2188697e-05, 3.1307377e-05, 2.0121186e-03, 2.0048373e-03, 1.8235679e-03]),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4])  # Only one solver since it is very long
def test_muscle_activations_with_contact_driven_ocp(ode_solver):
    # TODO: This test should be removed when DynamicsFcn.MUSCLE_ACTIVATIONS_AND_TORQUE_DRIVEN_WITH_CONTACT is
    # unitary tested

    # Load static_arm_with_contact
    bioptim_folder = TestUtils.bioptim_folder()
    static_arm = TestUtils.load_module(bioptim_folder + "/examples/muscle_driven_ocp/static_arm_with_contact.py")
    ode_solver = ode_solver()

    ocp = static_arm.prepare_ocp(
        bioptim_folder + "/examples/muscle_driven_ocp/arm26_with_contact.bioMod",
        final_time=2,
        n_shooting=10,
        weight=1,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14351397970185203)

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (60, 1))
        np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

        # Check some of the results
        q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.0081671, -0.94509584, 3.07047323]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.00093981, 0.41157421, -0.55870943]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.49332839e-07, 1.47494809e-03, 5.20721575e-03]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-2.72476211e-06, -2.79524486e-04, 6.92600551e-04]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.29081617e-06, 1.64961906e-01, 9.99986809e-02, 4.01995665e-06, 4.13036938e-06, 1.03940164e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25988708e-03, 3.21882769e-05, 3.13076618e-05, 2.01160287e-03, 2.00431774e-03, 1.82289866e-03]),
        )

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.14350699571954104)

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (60, 1))
        np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

        # Check some of the results
        q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.00816709, -0.94509077, 3.07046606]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.00093983, 0.411599, -0.55874465]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.77284867e-07, 1.47710422e-03, 5.20766354e-03]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-2.72484502e-06, -2.79525145e-04, 6.92616311e-04]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.28911678e-06, 1.64996819e-01, 1.00010798e-01, 4.01956674e-06, 4.12996816e-06, 1.03949142e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.25994595e-03, 3.21879960e-05, 3.13075455e-05, 2.01165125e-03, 2.00436616e-03, 1.82298538e-03]),
        )

    else:
        # Check objective function value
        f = np.array(sol.cost)
        np.testing.assert_equal(f.shape, (1, 1))
        np.testing.assert_almost_equal(f[0, 0], 0.1435025030068162)

        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (60, 1))
        np.testing.assert_almost_equal(g, np.zeros((60, 1)), decimal=6)

        # Check some of the results
        q, qdot, tau, mus = sol.states["q"], sol.states["qdot"], sol.controls["tau"], sol.controls["muscles"]

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0, 0.07, 1.4]))
        np.testing.assert_almost_equal(q[:, -1], np.array([0.0081671, -0.9450881, 3.0704626]))
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0, 0.0, 0.0]))
        np.testing.assert_almost_equal(qdot[:, -1], np.array([0.0009398, 0.4116121, -0.5587618]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-3.9652660e-07, 1.4785825e-03, 5.2079505e-03]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-2.7248808e-06, -2.7952503e-04, 6.9262306e-04]))
        np.testing.assert_almost_equal(
            mus[:, 0],
            np.array([2.2873915e-06, 1.6502014e-01, 1.0001872e-01, 4.0192359e-06, 4.1296273e-06, 1.0395487e-01]),
        )
        np.testing.assert_almost_equal(
            mus[:, -1],
            np.array([4.2599697e-03, 3.2187363e-05, 3.1307175e-05, 2.0116712e-03, 2.0043861e-03, 1.8230214e-03]),
        )

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)


@pytest.mark.parametrize("ode_solver", [OdeSolver.RK4])  # Only one solver since it is very long
def test_muscle_excitation_with_contact_driven_ocp(ode_solver):
    # Load contact_forces_inequality_constraint_muscle_excitations
    bioptim_folder = TestUtils.bioptim_folder()
    contact = TestUtils.load_module(
        bioptim_folder
        + "/examples/muscle_driven_with_contact/contact_forces_inequality_constraint_muscle_excitations.py"
    )
    boundary = 50
    ode_solver = ode_solver()

    ocp = contact.prepare_ocp(
        bioptim_folder + "/examples/muscle_driven_with_contact/2segments_4dof_2contacts_1muscle.bioMod",
        phase_time=0.3,
        n_shooting=10,
        min_bound=boundary,
        ode_solver=ode_solver,
    )
    sol = ocp.solve()

    # Check objective function value
    f = np.array(sol.cost)
    np.testing.assert_equal(f.shape, (1, 1))
    np.testing.assert_almost_equal(f[0, 0], 0.14525619)

    # Check some of the results
    q, qdot, mus_states, tau, mus_controls = (
        sol.states["q"],
        sol.states["qdot"],
        sol.states["muscles"],
        sol.controls["tau"],
        sol.controls["muscles"],
    )

    if isinstance(ode_solver, OdeSolver.IRK):
        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (110, 1))
        np.testing.assert_almost_equal(g[:90], np.zeros((90, 1)), decimal=6)
        np.testing.assert_array_less(-g[90:], -boundary)
        expected_pos_g = np.array(
            [
                [51.5414325],
                [52.77742181],
                [57.57780262],
                [62.62940016],
                [65.1683722],
                [66.33551167],
                [65.82614885],
                [63.06016376],
                [57.23683342],
                [50.47124118],
                [156.35594176],
                [136.1362431],
                [89.86994764],
                [63.41325331],
                [57.493027],
                [55.09716611],
                [53.77813649],
                [52.90987628],
                [52.19502561],
                [50.56093511],
            ]
        )
        np.testing.assert_almost_equal(g[90:], expected_pos_g)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
        np.testing.assert_almost_equal(
            q[:, -1], np.array([-3.40708085e-01, 1.34155553e-01, -2.22589697e-04, 2.22589697e-04])
        )
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array([-2.01858700e00, 4.49316671e-04, 4.03717411e00, -4.03717411e00])
        )
        # initial and final muscle state
        np.testing.assert_almost_equal(mus_states[:, 0], np.array([0.5]))
        np.testing.assert_almost_equal(mus_states[:, -1], np.array([0.52946019]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-54.08860398]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-26.70209712]))
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.48071638]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.40159522]))

    elif isinstance(ode_solver, OdeSolver.RK8):
        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (110, 1))
        np.testing.assert_almost_equal(g[:90], np.zeros((90, 1)), decimal=6)
        np.testing.assert_array_less(-g[90:], -boundary)
        expected_pos_g = np.array(
            [
                [51.54108548],
                [52.77720093],
                [57.5776414],
                [62.62966321],
                [65.16873337],
                [66.33594321],
                [65.82669791],
                [63.06102595],
                [57.23848183],
                [50.47112677],
                [156.35763657],
                [136.13688244],
                [89.86990489],
                [63.41179686],
                [57.49195628],
                [55.09640086],
                [53.77757475],
                [52.9094631],
                [52.19492485],
                [50.56081268],
            ]
        )
        np.testing.assert_almost_equal(g[90:], expected_pos_g)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
        np.testing.assert_almost_equal(
            q[:, -1], np.array([-3.40708085e-01, 1.34155553e-01, -2.22589697e-04, 2.22589697e-04])
        )
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array([-2.01866580e00, 4.49415846e-04, 4.03733171e00, -4.03733171e00])
        )
        # initial and final muscle state
        np.testing.assert_almost_equal(mus_states[:, 0], np.array([0.5]))
        np.testing.assert_almost_equal(mus_states[:, -1], np.array([0.5289569]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-54.0891972]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-26.7018241]))
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.4808524]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.4007721]))

    else:
        # Check constraints
        g = np.array(sol.constraints)
        np.testing.assert_equal(g.shape, (110, 1))
        np.testing.assert_almost_equal(g[:90], np.zeros((90, 1)))
        np.testing.assert_array_less(-g[90:], -boundary)
        expected_pos_g = np.array(
            [
                [51.5673555],
                [52.82179693],
                [57.5896514],
                [62.60246484],
                [65.13414631],
                [66.29498636],
                [65.77592127],
                [62.98288508],
                [57.0934291],
                [50.47918162],
                [156.22933663],
                [135.96633458],
                [89.93755291],
                [63.57705684],
                [57.59613028],
                [55.17020948],
                [53.83337907],
                [52.95213608],
                [52.20317604],
                [50.57048159],
            ]
        )
        np.testing.assert_almost_equal(g[90:], expected_pos_g, decimal=6)

        # initial and final position
        np.testing.assert_almost_equal(q[:, 0], np.array([0.0, 0.0, -0.75, 0.75]))
        np.testing.assert_almost_equal(
            q[:, -1], np.array([-3.40710032e-01, 1.34155565e-01, -2.18684502e-04, 2.18684502e-04])
        )
        # initial and final velocities
        np.testing.assert_almost_equal(qdot[:, 0], np.array([0.0, 0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(
            qdot[:, -1], np.array([-2.01607708e00, 4.40761528e-04, 4.03215433e00, -4.03215433e00])
        )
        # initial and final muscle state
        np.testing.assert_almost_equal(mus_states[:, 0], np.array([0.5]))
        np.testing.assert_almost_equal(mus_states[:, -1], np.array([0.54388439]))
        # initial and final controls
        np.testing.assert_almost_equal(tau[:, 0], np.array([-54.04429218]))
        np.testing.assert_almost_equal(tau[:, -1], np.array([-26.70770378]))
        np.testing.assert_almost_equal(mus_controls[:, 0], np.array([0.47810392]))
        np.testing.assert_almost_equal(mus_controls[:, -1], np.array([0.42519766]))

    # save and load
    TestUtils.save_and_load(sol, ocp, False)

    # simulate
    TestUtils.simulate(sol)
