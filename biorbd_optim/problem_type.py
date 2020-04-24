from casadi import MX, vertcat
import numpy as np

from .dynamics import Dynamics
from .mapping import BidirectionalMapping, Mapping


class ProblemType:
    """
    Includes methods suitable for several situations
    """

    @staticmethod
    def torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques but without muscles, must be used with dynamics without contacts.
        :param nlp: An instance of the OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_driven
        ProblemType.__configure_torque_driven(nlp)

    @staticmethod
    def torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques, without muscles, must be used with dynamics with contacts.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_driven_with_contact
        ProblemType.__configure_torque_driven(nlp)

    @staticmethod
    def __configure_torque_driven(nlp):
        """
        Configures common settings for torque driven problems with and without contacts.
        :param nlp: An OptimalControlProgram class.
        """
        if nlp["q_mapping"] is None:
            nlp["q_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbQ())), Mapping(range(nlp["model"].nbQ()))
            )
        if nlp["q_dot_mapping"] is None:
            nlp["q_dot_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbQdot())), Mapping(range(nlp["model"].nbQdot()))
            )
        if nlp["tau_mapping"] is None:
            nlp["tau_mapping"] = BidirectionalMapping(
                Mapping(range(nlp["model"].nbGeneralizedTorque())), Mapping(range(nlp["model"].nbGeneralizedTorque()))
            )

        dof_names = nlp["model"].nameDof()
        q = MX()
        q_dot = MX()
        for i in nlp["q_mapping"].reduce.map_idx:
            q = vertcat(q, MX.sym("Q_" + dof_names[i].to_string()))
        for i in nlp["q_dot_mapping"].reduce.map_idx:
            q_dot = vertcat(q_dot, MX.sym("Qdot_" + dof_names[i].to_string()))
        nlp["x"] = vertcat(q, q_dot)

        u = MX()
        for i in nlp["tau_mapping"].reduce.map_idx:
            u = vertcat(u, MX.sym("Tau_" + dof_names[i].to_string()))
        nlp["u"] = u

        nlp["nx"] = nlp["x"].rows()
        nlp["nu"] = nlp["u"].rows()

        nlp["nbQ"] = nlp["q_mapping"].reduce.len
        nlp["nbQdot"] = nlp["q_dot_mapping"].reduce.len
        nlp["nbTau"] = nlp["tau_mapping"].reduce.len
        nlp["nbMuscle"] = 0

    @staticmethod
    def muscles_and_torque_driven(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_muscle_driven
        ProblemType.__configure_torque_driven(nlp)

        u = MX()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["model"].nbMuscleTotal()):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)
        nlp["nu"] = nlp["u"].rows()

        nlp["nbMuscle"] = nlp["model"].nbMuscleTotal()

    @staticmethod
    def muscles_and_torque_driven_with_contact(nlp):
        """
        Names states (nlp.x) and controls (nlp.u) and gives size to (nlp.nx) and (nlp.nu).
        Works with torques and muscles.
        :param nlp: An OptimalControlProgram class.
        """
        nlp["dynamics_func"] = Dynamics.forward_dynamics_torque_muscle_driven_with_contact
        ProblemType.__configure_torque_driven(nlp)

        u = MX()
        muscle_names = nlp["model"].muscleNames()
        for i in range(nlp["model"].nbMuscleTotal()):
            u = vertcat(u, MX.sym("Muscle_" + muscle_names[i].to_string() + "_activation"))
        nlp["u"] = vertcat(nlp["u"], u)

        nlp["nu"] = nlp["u"].rows()

        nlp["nbMuscle"] = nlp["model"].nbMuscleTotal()

    @staticmethod
    def get_data_from_V_phase(V_phase, var_size, nb_nodes, offset, nb_variables, duplicate_last_column):
        """
        Extracts variables from V.
        :param V_phase: numpy array : Extract of V for a phase.
        """
        array = np.ndarray((var_size, nb_nodes))
        for dof in range(var_size):
            array[dof] = V_phase[offset + dof :: nb_variables]

        if duplicate_last_column:
            return np.c_[array, array[:, -1]]
        else:
            return array

    @staticmethod
    def get_data_from_V(ocp, V, num_phase=None):
        V_array = np.array(V).squeeze()
        has_muscles = False

        if num_phase is None:
            num_phase = range(len(ocp.nlp))
        elif isinstance(num_phase, int):
            num_phase = [num_phase]
        offsets = [0]
        for i, nlp in enumerate(ocp.nlp):
            offsets.append(offsets[i] + nlp["nx"] * (nlp["ns"] + 1) + nlp["nu"] * (nlp["ns"]))

        q, q_dot, tau, muscle = [], [], [], []

        for i in num_phase:
            nlp = ocp.nlp[i]

            V_phase = np.array(V_array[offsets[i] : offsets[i + 1]])
            nb_var = nlp["nx"] + nlp["nu"]

            if (
                nlp["problem_type"] == ProblemType.torque_driven
                or nlp["problem_type"] == ProblemType.torque_driven_with_contact
                or nlp["problem_type"] == ProblemType.muscles_and_torque_driven
                or nlp["problem_type"] == ProblemType.muscles_and_torque_driven_with_contact
            ):
                q.append(ProblemType.get_data_from_V_phase(V_phase, nlp["nbQ"], nlp["ns"] + 1, 0, nb_var, False))
                q_dot.append(
                    ProblemType.get_data_from_V_phase(V_phase, nlp["nbQdot"], nlp["ns"] + 1, nlp["nbQ"], nb_var, False)
                )
                tau.append(ProblemType.get_data_from_V_phase(V_phase, nlp["nbTau"], nlp["ns"], nlp["nx"], nb_var, True))

                if (
                    nlp["problem_type"] == ProblemType.muscles_and_torque_driven
                    or nlp["problem_type"] == ProblemType.muscles_and_torque_driven_with_contact
                ):
                    has_muscles = True
                    muscle.append(
                        ProblemType.get_data_from_V_phase(
                            V_phase, nlp["nbMuscle"], nlp["ns"], nlp["nx"] + nlp["nbTau"], nb_var, True,
                        )
                    )
                else:
                    muscle.append([])

            else:
                raise RuntimeError(f"{nlp['problem_type'].__name__} not implemented yet in get_data_from_V")

        if len(num_phase) == 1:
            q = q[0]
            q_dot = q_dot[0]
            tau = tau[0]
            muscle = muscle[0]
        if has_muscles:
            return q, q_dot, tau, muscle
        else:
            return q, q_dot, tau

    @staticmethod
    def get_q_from_V(ocp, V, num_phase=None):
        if ocp.nlp[0]["problem_type"] == ProblemType.torque_driven:
            x, _, _ = ProblemType.get_data_from_V(ocp, V, num_phase)

        elif (
            ocp.nlp[0]["problem_type"] == ProblemType.muscles_and_torque_driven
            or ocp.nlp[0]["problem_type"] == ProblemType.muscles_and_torque_driven_with_contact
        ):
            x, _, _, _ = ProblemType.get_data_from_V(ocp, V, num_phase)

        else:
            raise RuntimeError(f"{ocp.nlp[0]['problem_type']} is not implemented for this type of OCP")
        return x
