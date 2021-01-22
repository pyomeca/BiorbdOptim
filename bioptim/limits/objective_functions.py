from typing import Callable, Union
from enum import Enum

import numpy as np
from casadi import MX, SX

from .penalty import PenaltyType, PenaltyFunctionAbstract, PenaltyOption
from ..misc.enums import Node
from ..misc.options import OptionList, OptionGeneric


class Objective(PenaltyOption):
    """
    A placeholder for an objective function

    Attributes
    ----------
    weight: float
        The weighting applied to this specific objective function
    """

    def __init__(self, objective, weight: float = 1, custom_type=None, phase: int = 0, **params):
        """
        Parameters
        ----------
        objective: Union[ObjectiveFcn.Lagrange, ObjectiveFcn.Mayer, Callable[OptimalControlProgram, MX]]
            The chosen objective function
        weight: float
            The weighting applied to this specific objective function
        custom_type: Objective
            When objective is a custom defined function, one must specify if the custom_type is Mayer or Lagrange
        phase: int
            At which phase this objective function must be applied
        params: dict
            Generic parameters for options
        """

        custom_function = None
        if not isinstance(objective, ObjectiveFcn.Lagrange) and not isinstance(objective, ObjectiveFcn.Mayer):
            custom_function = objective

            if custom_type is None:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is missing. "
                    "It should either be ObjectiveFcn.Mayer or ObjectiveFcn.Lagrange"
                )
            objective = custom_type(custom_type.CUSTOM)
            if isinstance(objective, ObjectiveFcn.Lagrange):
                pass
            elif isinstance(objective, ObjectiveFcn.Mayer):
                pass
            elif isinstance(objective, ObjectiveFcn.Parameter):
                pass
            else:
                raise RuntimeError(
                    "Custom objective function detected, but custom_function is invalid. "
                    "It should either be ObjectiveFcn.Mayer or ObjectiveFcn.Lagrange"
                )

        super(Objective, self).__init__(penalty=objective, phase=phase, custom_function=custom_function, **params)
        self.weight = weight


class ObjectiveList(OptionList):
    """
    A list of Constraint if more than one is required

    Methods
    -------
    add(self, constraint: Union[Callable, "ConstraintFcn"], **extra_arguments)
        Add a new Constraint to the list
    """

    def add(self, objective: Union[Callable, Objective], **extra_arguments):
        """
        Add a new objective function to the list

        Parameters
        ----------
        objective: Union[Callable, Objective]
            The chosen objective function
        extra_arguments: dict
            Any parameters to pass to ObjectiveFcn
        """

        if isinstance(objective, Objective):
            self.copy(objective)
        else:
            super(ObjectiveList, self)._add(option_type=Objective, objective=objective, **extra_arguments)


class ObjectiveFunction:
    """
    Internal (re)implementation of the penalty functions

    Methods
    -------
    add_or_replace(ocp: OptimalControlProgram, nlp: NonLinearProgram, objective: Objective)
        Add the objective function to the objective pool
    add_to_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, val: Union[MX, SX], penalty: Objective, dt:float=0)
        Add the objective function to the objective pool
    clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Objective)
        Resets a objective function. A negative penalty index creates a new empty objective function.
    """

    class LagrangeFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions

        Methods
        -------
        add_to_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, val: Union[MX, SX], penalty: Objective)
            Add the objective function to the objective pool
        clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Objective)
            Resets a objective function. A negative penalty index creates a new empty objective function.
        _parameter_modifier(objective: Objective)
            Apply some default parameters
        _span_checker(objective: Objective, nlp: NonLinearProgram)
            Check for any non sense in the requested times for the constraint. Raises an error if so
        penalty_nature() -> str
            Get the nature of the penalty
        """

        class Functions:
            """
            Implementation of all the Lagrange objective functions

            Methods
            -------
            minimize_time(penalty: "ObjectiveFcn.Lagrange", ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
                Minimizes the duration of the phase
            """

            @staticmethod
            def minimize_time(
                penalty: Objective,
                ocp,
                nlp,
                t: list,
                x: list,
                u: list,
                p: Union[MX, SX],
            ):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                penalty: Objective,
                    The actual constraint to declare
                ocp: OptimalControlProgram
                    A reference to the ocp
                nlp: NonLinearProgram
                    A reference to the current phase of the ocp
                t: list
                    Time indices, maximum value being the number of shooting point + 1
                x: list
                    References to the state variables
                u: list
                    References to the control variables
                p: Union[MX, SX]
                    References to the parameter variables
                """

                val = 1
                ObjectiveFunction.LagrangeFunction.add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def add_to_penalty(ocp, nlp, val: Union[MX, SX], penalty: Objective):
            """
            Add the objective function to the objective pool

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            val: Union[MX, SX]
                The actual objective function to add
            penalty: Objective
                The actual objective function to declare
            """

            ObjectiveFunction.add_to_penalty(ocp, nlp, val, penalty, dt=nlp.dt)

        @staticmethod
        def clear_penalty(ocp, nlp, penalty: Objective):
            """
            Resets a objective function. A negative penalty index creates a new empty objective function.

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            penalty: Objective
                The actual objective function to declare
            """

            return ObjectiveFunction.clear_penalty(ocp, nlp, penalty)

        @staticmethod
        def _parameter_modifier(objective: Objective):
            """
            Apply some default parameters

            Parameters
            ----------
            objective: Objective
                The actual objective function to declare
            """

            func = objective.type.value[0]
            # Everything that should change the entry parameters depending on the penalty can be added here
            if func == ObjectiveFcn.Lagrange.MINIMIZE_TIME.value[0]:
                # max_bound ans min_bound are already dealt with in OptimalControlProgram.__define_parameters_phase_time
                if "min_bound" in objective.params:
                    raise RuntimeError(
                        "ObjectiveFcn.Lagrange.MINIMIZE_TIME cannot have min_bound. "
                        "Please either use MAYER or constraint"
                    )
                if "max_bound" in objective.params:
                    raise RuntimeError(
                        "ObjectiveFcn.Lagrange.MINIMIZE_TIME cannot have max_bound. "
                        "Please either use MAYER or constraint"
                    )
                if not objective.quadratic:
                    objective.quadratic = True
            PenaltyFunctionAbstract._parameter_modifier(objective)

        @staticmethod
        def _span_checker(objective: Objective, nlp):
            """
            Check for any non sense in the requested times for the constraint. Raises an error if so

            Parameters
            ----------
            objective: Objective
                The actual objective function to declare
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            """

            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(objective, nlp)

        @staticmethod
        def add_or_replace(ocp, nlp, objective: PenaltyOption):
            """
            Add the objective function to the objective pool

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            objective: PenaltyOption
                The actual objective function to declare
            """
            ObjectiveFunction.add_or_replace(ocp, nlp, objective)

        @staticmethod
        def penalty_nature() -> str:
            """
            Get the nature of the penalty

            Returns
            -------
            The nature of the penalty
            """

            return "objective_functions"

    class MayerFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions

        Methods
        -------
        inter_phase_continuity(ocp: OptimalControlProgram, pt: "StateTransition")
            Add phase transition objective between two phases.
        add_to_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, val: Union[MX, SX], penalty: Objective)
            Add the objective function to the objective pool
        clear_penalty(ocp: OptimalControlProgram, nlp: NonLinearProgram, penalty: Objective)
            Resets a objective function. A negative penalty index creates a new empty objective function.
        _parameter_modifier(objective: Objective)
            Apply some default parameters
        _span_checker(objective: Objective, nlp: NonLinearProgram)
            Check for any non sense in the requested times for the constraint. Raises an error if so
        penalty_nature() -> str
            Get the nature of the penalty
        """

        class Functions:
            """
            Implementation of all the Mayer objective functions

            Methods
            -------
            minimize_time(penalty: "ObjectiveFcn.Lagrange", ocp: OptimalControlProgram, nlp: NonLinearProgram, t: list, x: list, u: list, p: Union[MX, SX])
                Minimizes the duration of the phase
            """

            @staticmethod
            def minimize_time(
                penalty: Objective,
                ocp,
                nlp,
                t: list,
                x: list,
                u: list,
                p: Union[MX, SX],
            ):
                """
                Minimizes the duration of the phase

                Parameters
                ----------
                penalty: Objective,
                    The actual constraint to declare
                ocp: OptimalControlProgram
                    A reference to the ocp
                nlp: NonLinearProgram
                    A reference to the current phase of the ocp
                t: list
                    Time indices, maximum value being the number of shooting point + 1
                x: list
                    References to the state variables
                u: list
                    References to the control variables
                p: Union[MX, SX]
                    References to the parameter variables
                """

                val = nlp.tf
                ObjectiveFunction.MayerFunction.add_to_penalty(ocp, nlp, val, penalty)

        @staticmethod
        def inter_phase_continuity(ocp, pt: "StateTransition"):
            """
            Add phase transition objective between two phases.

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            pt: "StateTransition"
                The state transition to add
            """

            # Dynamics must be respected between phases
            penalty = OptionGeneric()
            penalty.list_index = -1
            penalty.quadratic = pt.quadratic
            penalty.weight = pt.weight
            penalty.sliced_target = None
            pt.base.clear_penalty(ocp, None, penalty)
            val = pt.type.value[0](ocp, pt)
            pt.base.add_to_penalty(ocp, None, val, penalty)

        @staticmethod
        def add_to_penalty(ocp, nlp, val: Union[MX, SX], penalty: Objective):
            """
            Add the objective function to the objective pool

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            val: Union[MX, SX]
                The actual objective function to add
            penalty: Objective
                The actual objective function to declare
            """

            ObjectiveFunction.add_to_penalty(ocp, nlp, val, penalty, dt=1)

        @staticmethod
        def clear_penalty(ocp, nlp, penalty: Objective):
            """
            Resets a objective function. A negative penalty index creates a new empty objective function.

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            penalty: Objective
                The actual objective function to declare
            """

            return ObjectiveFunction.clear_penalty(ocp, nlp, penalty)

        @staticmethod
        def _parameter_modifier(objective: Objective):
            """
            Apply some default parameters

            Parameters
            ----------
            objective: Objective
                The actual objective function to declare
            """

            func = objective.type.value[0]
            # Everything that should change the entry parameters depending on the penalty can be added here
            if func == ObjectiveFcn.Mayer.MINIMIZE_TIME.value[0]:
                # max_bound ans min_bound are already dealt with in OptimalControlProgram.__define_parameters_phase_time
                if "min_bound" in objective.params:
                    del objective.params["min_bound"]
                if "max_bound" in objective.params:
                    del objective.params["max_bound"]

            PenaltyFunctionAbstract._parameter_modifier(objective)

        @staticmethod
        def _span_checker(objective: Objective, nlp):
            """
            Check for any non sense in the requested times for the constraint. Raises an error if so

            Parameters
            ----------
            objective: Objective
                The actual objective function to declare
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            """

            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(objective, nlp)

        @staticmethod
        def add_or_replace(ocp, nlp, objective: PenaltyOption):
            """
            Add the objective function to the objective pool

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            objective: PenaltyOption
                The actual objective function to declare
            """
            ObjectiveFunction.add_or_replace(ocp, nlp, objective)

        @staticmethod
        def penalty_nature() -> str:
            """
            Get the nature of the penalty

            Returns
            -------
            The nature of the penalty
            """

            return "objective_functions"

    class ParameterFunction(PenaltyFunctionAbstract):
        """
        Internal (re)implementation of the penalty functions

        add_to_penalty(ocp: OptimalControlProgram, _, val: Union[MX, SX], penalty: Objective)
            Add the objective function to the objective pool
        clear_penalty(ocp: OptimalControlProgram, _, penalty: Objective)
            Resets a objective function. A negative penalty index creates a new empty objective function.
        _parameter_modifier(objective: Objective)
            Apply some default parameters
        _span_checker(objective: Objective, nlp: NonLinearProgram)
            Check for any non sense in the requested times for the constraint. Raises an error if so
        penalty_nature() -> str
            Get the nature of the penalty
        """

        class Functions:
            """
            Implementation of all the parameters objective functions
            """

            pass

        @staticmethod
        def add_to_penalty(ocp, _, val: Union[MX, SX], penalty: Objective):
            """
            Add the objective function to the objective pool

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            val: Union[MX, SX]
                The actual objective function to add
            penalty: Objective
                The actual objective function to declare
            """
            ObjectiveFunction.add_to_penalty(ocp, None, val, penalty, dt=1)

        @staticmethod
        def clear_penalty(ocp, _, penalty: Objective):
            """
            Resets a objective function. A negative penalty index creates a new empty objective function.

            Parameters
            ----------
            ocp: OptimalControlProgram
                A reference to the ocp
            penalty: Objective
                The actual objective function to declare
            """

            return ObjectiveFunction.clear_penalty(ocp, None, penalty)

        @staticmethod
        def _parameter_modifier(objective: Objective):
            """
            Apply some default parameters

            Parameters
            ----------
            objective: Objective
                The actual objective function to declare
            """

            # Everything that should change the entry parameters depending on the penalty can be added here
            PenaltyFunctionAbstract._parameter_modifier(objective)

        @staticmethod
        def _span_checker(objective: Objective, nlp):
            """
            Check for any non sense in the requested times for the constraint. Raises an error if so

            Parameters
            ----------
            objective: Objective
                The actual objective function to declare
            nlp: NonLinearProgram
                A reference to the current phase of the ocp
            """

            # Everything that is suspicious in terms of the span of the penalty function ca be checked here
            PenaltyFunctionAbstract._span_checker(objective, nlp)

        @staticmethod
        def penalty_nature() -> str:
            """
            Get the nature of the penalty

            Returns
            -------
            The nature of the penalty
            """

            return "parameters"

    @staticmethod
    def add_or_replace(ocp, nlp, objective: PenaltyOption):
        """
        Add the objective function to the objective pool

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        objective: PenaltyOption
            The actual objective function to declare
        """

        if objective.type.get_type() == ObjectiveFunction.LagrangeFunction:
            if objective.node != Node.ALL and objective.node != Node.DEFAULT:
                raise RuntimeError("Lagrange objective are for Node.ALL, did you mean Mayer?")
            objective.node = Node.ALL
        elif objective.type.get_type() == ObjectiveFunction.MayerFunction:
            if objective.node == Node.DEFAULT:
                objective.node = Node.END

        else:
            raise RuntimeError("ObjectiveFcn function Type must be either a Lagrange or Mayer type")
        PenaltyFunctionAbstract.add_or_replace(ocp, nlp, objective)

    @staticmethod
    def add_to_penalty(ocp, nlp, val: Union[MX, SX], penalty: Objective, dt: float = 0):
        """
        Add the objective function to the objective pool

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        val: Union[MX, SX]
            The actual objective function to add
        penalty: Objective
            The actual objective function to declare
        dt: float
            The time between two nodes for the current phase. If the objective is Mayer, dt should be 1
        """

        J = {"objective": penalty, "val": val, "target": penalty.sliced_target, "dt": dt}

        if nlp:
            nlp.J[penalty.list_index].append(J)
        else:
            ocp.J[penalty.list_index].append(J)

    @staticmethod
    def clear_penalty(ocp, nlp, penalty: Objective):
        """
        Resets a objective function. A negative penalty index creates a new empty objective function.

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        nlp: NonLinearProgram
            A reference to the current phase of the ocp
        penalty: Objective
            The actual objective function to declare
        """

        if nlp:
            J_to_add_to = nlp.J
        else:
            J_to_add_to = ocp.J

        if penalty.list_index < 0:
            # Add a new one
            for i, j in enumerate(J_to_add_to):
                if not j:
                    penalty.list_index = i
                    return
            else:
                J_to_add_to.append([])
                penalty.list_index = len(J_to_add_to) - 1
        else:
            while penalty.list_index >= len(J_to_add_to):
                J_to_add_to.append([])
            J_to_add_to[penalty.list_index] = []


class ObjectivePrinter:
    """
    Interface to print the values of the objectives to the console

    Methods
    -------


    """

    def __init__(self, ocp, sol_obj: dict):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp
        sol_obj: dict
            A reference to the solution structure
        """

        self.ocp = ocp
        self.sol_obj = sol_obj

    def by_function(self):
        """
        Print the values of the objectives sorted by function
        """

        for idx_phase, phase in enumerate(self.sol_obj):
            print(f"********** Phase {idx_phase} **********")
            for idx_obj in range(phase.shape[0]):
                print(
                    f"{self.ocp.original_values['objective_functions'][idx_phase][idx_phase + idx_obj].type.name} : {np.nansum(phase[idx_obj])}"
                )

    def by_nodes(self):
        """
        Print the values of the objectives sorted by time node
        """

        for idx_phase, phase in enumerate(self.sol_obj):
            print(f"********** Phase {idx_phase} **********")
            for idx_node in range(phase.shape[1]):
                print(f"Node {idx_node} : {np.nansum(phase[:, idx_node])}")

    def mean(self):
        """
        Get the mean of the objective functions

        Returns
        """

        m = 0
        for idx_phase, phase in enumerate(self.sol_obj):
            m += np.nansum(phase)
        return m / len(self.sol_obj)


class ObjectiveFcn:
    """
    Selection of valid objective functions
    """

    class Lagrange(Enum):
        """
        Selection of valid Lagrange objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        MINIMIZE_TIME = (ObjectiveFunction.LagrangeFunction.Functions.minimize_time,)
        MINIMIZE_STATE = (PenaltyType.MINIMIZE_STATE,)
        TRACK_STATE = (PenaltyType.TRACK_STATE,)
        MINIMIZE_MARKERS = (PenaltyType.MINIMIZE_MARKERS,)
        TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
        MINIMIZE_MARKERS_DISPLACEMENT = (PenaltyType.MINIMIZE_MARKERS_DISPLACEMENT,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyType.MINIMIZE_MARKERS_VELOCITY,)
        TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
        FOLLOW_MARKERS = (PenaltyType.FOLLOW_MARKERS,)
        PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
        PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
        MINIMIZE_TORQUE = (PenaltyType.MINIMIZE_TORQUE,)
        TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
        MINIMIZE_TORQUE_DERIVATIVE = (PenaltyType.MINIMIZE_TORQUE_DERIVATIVE,)
        MINIMIZE_MUSCLES_CONTROL = (PenaltyType.MINIMIZE_MUSCLES_CONTROL,)
        TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
        MINIMIZE_ALL_CONTROLS = (PenaltyType.MINIMIZE_ALL_CONTROLS,)
        TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
        MINIMIZE_CONTACT_FORCES = (PenaltyType.MINIMIZE_CONTACT_FORCES,)
        TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
        MINIMIZE_COM_POSITION = (PenaltyType.MINIMIZE_COM_POSITION,)
        MINIMIZE_COM_VELOCITY = (PenaltyType.MINIMIZE_COM_VELOCITY,)
        ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
        ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.LagrangeFunction

    class Mayer(Enum):
        """
        Selection of valid Mayer objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        MINIMIZE_TIME = (ObjectiveFunction.MayerFunction.Functions.minimize_time,)
        MINIMIZE_STATE = (PenaltyType.MINIMIZE_STATE,)
        TRACK_STATE = (PenaltyType.TRACK_STATE,)
        MINIMIZE_MARKERS = (PenaltyType.MINIMIZE_MARKERS,)
        TRACK_MARKERS = (PenaltyType.TRACK_MARKERS,)
        MINIMIZE_MARKERS_DISPLACEMENT = (PenaltyType.MINIMIZE_MARKERS_DISPLACEMENT,)
        MINIMIZE_MARKERS_VELOCITY = (PenaltyType.MINIMIZE_MARKERS_VELOCITY,)
        TRACK_MARKERS_VELOCITY = (PenaltyType.TRACK_MARKERS_VELOCITY,)
        FOLLOW_MARKERS = (PenaltyType.FOLLOW_MARKERS,)
        PROPORTIONAL_STATE = (PenaltyType.PROPORTIONAL_STATE,)
        PROPORTIONAL_CONTROL = (PenaltyType.PROPORTIONAL_CONTROL,)
        MINIMIZE_TORQUE = (PenaltyType.MINIMIZE_TORQUE,)
        TRACK_TORQUE = (PenaltyType.TRACK_TORQUE,)
        MINIMIZE_MUSCLES_CONTROL = (PenaltyType.MINIMIZE_MUSCLES_CONTROL,)
        TRACK_MUSCLES_CONTROL = (PenaltyType.TRACK_MUSCLES_CONTROL,)
        MINIMIZE_ALL_CONTROLS = (PenaltyType.MINIMIZE_ALL_CONTROLS,)
        TRACK_ALL_CONTROLS = (PenaltyType.TRACK_ALL_CONTROLS,)
        MINIMIZE_CONTACT_FORCES = (PenaltyType.MINIMIZE_CONTACT_FORCES,)
        TRACK_CONTACT_FORCES = (PenaltyType.TRACK_CONTACT_FORCES,)
        MINIMIZE_PREDICTED_COM_HEIGHT = (PenaltyType.MINIMIZE_PREDICTED_COM_HEIGHT,)
        MINIMIZE_COM_POSITION = (PenaltyType.MINIMIZE_COM_POSITION,)
        MINIMIZE_COM_VELOCITY = (PenaltyType.MINIMIZE_COM_VELOCITY,)
        ALIGN_SEGMENT_WITH_CUSTOM_RT = (PenaltyType.ALIGN_SEGMENT_WITH_CUSTOM_RT,)
        ALIGN_MARKER_WITH_SEGMENT_AXIS = (PenaltyType.ALIGN_MARKER_WITH_SEGMENT_AXIS,)
        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.MayerFunction

    class Parameter(Enum):
        """
        Selection of valid Parameters objective functions

        Methods
        -------
        def get_type() -> Callable
            Returns the type of the penalty
        """

        CUSTOM = (PenaltyType.CUSTOM,)

        @staticmethod
        def get_type() -> Callable:
            """
            Returns the type of the penalty
            """
            return ObjectiveFunction.ParameterFunction
