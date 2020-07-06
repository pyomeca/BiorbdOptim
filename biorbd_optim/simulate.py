import numpy as np
from copy import copy


class Simulate:
    @staticmethod
    def from_solve(ocp, sol, single_shoot=False):
        v_input = np.array(sol["x"]).squeeze()
        v_output = copy(v_input)
        offset = 0
        for nlp in ocp.nlp:
            # TODO adds StateTransitionFunctions between phases
            for idx_nodes in range(nlp["ns"]):
                x0 = v_input[offset : offset + nlp["nx"]] if single_shoot else v_output[offset : offset + nlp["nx"]]
                v_output[offset + nlp["nx"] + nlp["nu"] : offset + 2 * nlp["nx"] + nlp["nu"]] = np.array(
                    nlp["dynamics"][idx_nodes](x0=x0, p=v_input[offset + nlp["nx"] : offset + nlp["nx"] + nlp["nu"]])[
                        "xf"
                    ]
                ).squeeze()
                offset += nlp["nx"] + nlp["nu"]
        sol["x"] = v_output
        return sol

    @staticmethod
    def from_data(ocp, data, single_shoot=True):
        states = data[0]
        controls = data[1]
        v = np.ndarray(0)

        offset_phases = 0
        for nlp in ocp.nlp:
            offset = 0
            v_phase = np.ndarray((nlp["ns"] + 1) * nlp["nx"] + nlp["ns"] * nlp["nu"])
            v_phase[offset : offset + nlp["nx"]] = Simulate._concat_variables(states, offset_phases, 0)
            for idx_nodes in range(nlp["ns"]):
                x0 = (
                    Simulate._concat_variables(states, offset_phases, idx_nodes)
                    if single_shoot
                    else v_phase[offset : offset + nlp["nx"]]
                )
                v_phase[offset + nlp["nx"] + nlp["nu"] : offset + 2 * nlp["nx"] + nlp["nu"]] = np.array(
                    nlp["dynamics"][idx_nodes](
                        x0=x0, p=Simulate._concat_variables(controls, offset_phases, idx_nodes),
                    )["xf"]
                ).squeeze()
                offset += nlp["nx"] + nlp["nu"]
            v = np.append(v, v_phase)
            offset_phases += nlp["ns"]
        return {"x": v}

    @staticmethod
    def from_controls_and_initial_states(ocp, states, controls, single_shoot=False):
        # todo flag single/multiple here and in from_solve (copy states)
        states.check_and_adjust_dimensions(ocp.nlp[0]["nx"], ocp.nlp[0]["ns"])
        v = states.init.evaluate_at(0)

        if not isinstance(controls, (list, tuple)):
            controls = (controls,)

        for idx_phase, nlp in enumerate(ocp.nlp):
            controls[idx_phase].check_and_adjust_dimensions(nlp["nu"], nlp["ns"] - 1)
            for idx_nodes in range(nlp["ns"]):
                v = np.append(v, controls[idx_phase].init.evaluate_at(shooting_point=idx_nodes))
                v = np.append(v, states.init.evaluate_at(0))

        return Simulate.from_solve(ocp, {"x": v}, single_shoot)

    @staticmethod
    def _concat_variables(variables, offset_phases, idx_nodes):
        var = np.ndarray(0)
        for key in variables.keys():
            var = np.append(var, variables[key][:, offset_phases + idx_nodes])
        return var