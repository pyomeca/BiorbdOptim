from pendulum import prepare_ocp

from biorbd_optim import InitialConditions, Simulate, ShowResult

ocp = prepare_ocp(
    biorbd_model_path="pendulum.bioMod",
    final_time=2,
    number_shooting_points=10,
    nb_threads=2,
)

X = InitialConditions([0, 0, 0, 0])
U = InitialConditions([-1, 1])

# --- Single shooting --- #
sol_simulate_single_shooting = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=True)
result_single = ShowResult(ocp, sol_simulate_single_shooting)
result_single.graphs()

# --- Multiple shooting --- #
sol_simulate_multiple_shooting = Simulate.from_controls_and_initial_states(ocp, X, U, single_shoot=False)
result_multiple = ShowResult(ocp, sol_simulate_multiple_shooting)
result_multiple.graphs()

# --- Single shooting --- #
result_single.animate()

# --- Multiple shooting --- #
result_multiple.animate()