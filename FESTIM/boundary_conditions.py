import FESTIM
from fenics import *
import sympy as sp
import numpy as np


def define_dirichlet_bcs_T(simulation):
    """Creates a list of BCs for thermal problem

    Arguments:


    Returns:
        list -- contains fenics.DirichletBC
        list -- contains fenics.Expression to be updated
    """
    parameters = simulation.parameters
    V = simulation.V_CG1
    boundaries = simulation.surface_markers

    bcs = []
    expressions = []
    for bc in parameters["temperature"]["boundary_conditions"]:
        if bc["type"] == "dc":
            value = sp.printing.ccode(bc["value"])
            value = Expression(value, degree=2, t=0)
            expressions.append(value)
            if type(bc["surfaces"]) is list:
                surfaces = bc["surfaces"]
            else:
                surfaces = [bc["surfaces"]]
            for surf in surfaces:
                bci = DirichletBC(V, value, boundaries, surf)
                bcs.append(bci)
    return bcs, expressions


def apply_fluxes(simulation):
    """Modifies the formulation and adds fluxes based
    on parameters in boundary_conditions

    Arguments:


    Keyword Arguments:
        S {fenics.UserExpression} -- solubility (default: {None})

    Raises:
        NameError: if boundary condition type is unknown

    Returns:
        fenics.Form() -- formulation for BCs
        list -- contains all the fenics.Expression() to be updated
    """
    parameters = simulation.parameters
    u = simulation.u
    v = simulation.v
    ds = simulation.ds
    T = simulation.T
    S = simulation.S

    expressions = []
    solutions = split(u)
    testfunctions = split(v)
    solute = solutions[0]
    test_solute = testfunctions[0]
    F = 0
    k_B = FESTIM.k_B
    boundary_conditions = parameters["boundary_conditions"]
    conservation_chemic_pot = False
    if S is not None:
        for mat in parameters["materials"]:
            if "S_0" in mat.keys() or "E_S" in mat.keys():
                conservation_chemic_pot = True
        if conservation_chemic_pot:
            solute = solute*S

    for bc in boundary_conditions:
        if bc["type"] not in FESTIM.helpers.bc_types["dc"]:
            if bc["type"] not in FESTIM.helpers.bc_types["neumann"] and \
               bc["type"] not in FESTIM.helpers.bc_types["robin"]:

                raise NameError(
                    "Unknown boundary condition type : " + bc["type"])
            if bc["type"] == "flux":
                flux = sp.printing.ccode(bc["value"])
                flux = Expression(flux, t=0,
                                  degree=2)
                expressions.append(flux)
            elif bc["type"] == "recomb":
                Kr = bc["Kr_0"]*exp(-bc["E_Kr"]/k_B/T)
                flux = -Kr*solute**bc["order"]

            if type(bc['surfaces']) is not list:
                surfaces = [bc['surfaces']]
            else:
                surfaces = bc['surfaces']
            for surf in surfaces:
                F += -test_solute*flux*ds(surf)
    return F, expressions


class BoundaryConditionTheta(UserExpression):
    """Creates an Expression for converting dirichlet bcs in the case
    of chemical potential conservation

    Args:
        UserExpression (fenics.UserExpression):
    """
    def __init__(self, bci, mesh, materials, vm, T, **kwargs):
        """initialisation

        Args:
            bci (fenics.Expression): value of BC
            mesh (fenics.mesh): mesh
            materials (list): contains dicts for materials
            vm (fenics.MeshFunction): volume markers
            T (fenics.Function): Temperature
        """
        super().__init__(kwargs)
        self._bci = bci
        self._mesh = mesh
        self._vm = vm
        self._T = T
        self._materials = materials

    def eval_cell(self, value, x, ufc_cell):
        cell = Cell(self._mesh, ufc_cell.index)
        subdomain_id = self._vm[cell]
        material = FESTIM.helpers.find_material_from_id(
            self._materials, subdomain_id)
        S_0 = material["S_0"]
        E_S = material["E_S"]
        value[0] = self._bci(x)/(S_0*exp(-E_S/FESTIM.k_B/self._T(x)))

    def value_shape(self):
        return ()


class BoundaryConditionRecomb(UserExpression):
    """Creates an Expression for converting implanted flux in surface
    concentration

    Args:
        UserExpression (fenics.UserExpression): value of surface concentration
    """
    def __init__(self, phi, R_p, K_0, E_K, D_0, E_D, T, **kwargs):
        """initialisation

        Args:
            phi (float): implanted particle flux
            R_p (float): implantation depth
            K_0 (float): Recombination coefficient pre-exponential factor
            E_K (float): Recombination energy
            D_0 (float): Diffusion coefficient pre-exponential factor
            E_D (float): Diffusion energy
            T (fenics.Function(), fenics.Expression()): Temperature
        """
        super().__init__(kwargs)
        self._phi = phi
        self._R_p = R_p

        self._D_0 = D_0
        self._E_D = E_D
        self._K_0 = K_0
        self._E_K = E_K

        self._T = T

    def eval(self, value, x):
        D = self._D_0*exp(-self._E_D/FESTIM.k_B/self._T(x))
        val = self._phi(x)*self._R_p(x)/D
        if self._K_0 is not None:  # non-instantaneous recomb
            K = self._K_0*exp(-self._E_K/FESTIM.k_B/self._T(x))
            val += (self._phi(x)/K)**0.5
        value[0] = val

    def value_shape(self):
        return ()


class BoundaryConditionSolubility(UserExpression):
    """Class based on UserExpression to create a Sievert law boundary
    condition where c = sqrt(P)*S(T)

    Args:
        UserExpression (fenics.UserExpression):
    """
    def __init__(self, S_0, E_S, pressure, T, **kwargs):
        """initialisation

        Args:
            S_0 (float): Solubility pre activation factor (m^-3 Pa^-0.5)
            E_S (float): Solubility activation energy (eV)
            pressure (fenics.Expression): Hydrogen pressure (Pa)
            T (fenics.Function): Temperature (K)
        """
        super().__init__(kwargs)
        self._pressure = pressure

        self._S_0 = S_0
        self._E_S = E_S

        self._T = T

    def eval(self, value, x):
        S = self._S_0*exp(-self._E_S/FESTIM.k_B/self._T(x))
        val = S*self._pressure(x)**0.5
        value[0] = val

    def value_shape(self):
        return ()


def apply_boundary_conditions(simulation):
    """Create a list of DirichletBCs.

    Arguments:


    Raises:
        KeyError: Raised if the type key of bc is missing
        NameError: Raised if type is unknown

    Returns:
        list -- contains fenics DirichletBC
        list -- contains the fenics.Expression() to be updated
    """
    parameters = simulation.parameters
    V = simulation.V
    boundary_conditions = parameters["boundary_conditions"]
    volume_markers = simulation.volume_markers
    surface_markers = simulation.surface_markers
    T = simulation.T

    bcs = list()
    expressions = list()

    for BC in boundary_conditions:
        if "type" in BC.keys():
            type_BC = BC["type"]
        else:
            raise KeyError("Missing boundary condition type key")
        if type_BC == "dc":
            value_BC = sp.printing.ccode(BC['value'])
            value_BC = Expression(value_BC, t=0, degree=4)
        elif type_BC == "solubility":
            # create a time dependent expression for pressure
            pressure = Expression(sp.printing.ccode(BC["pressure"]), t=0,
                                  degree=2)
            expressions.append(pressure)  # add it to the list of exprs

            # create a custom expression
            value_BC = BoundaryConditionSolubility(
                BC["S_0"], BC["E_S"],
                pressure, T)
        elif type_BC == "dc_imp":
            # Create 2 Expressions for phi and R_p
            phi = Expression(sp.printing.ccode(BC["implanted_flux"]),
                             t=0,
                             degree=1)
            R_p = Expression(sp.printing.ccode(BC["implantation_depth"]),
                             t=0,
                             degree=1)
            expressions.append(phi)  # add to the expressions to be updated
            expressions.append(R_p)
            D_0, E_D = BC["D_0"], BC["E_D"]
            K_0, E_K = None, None  # instantaneous recomb
            if "K_0" in BC.keys() and "E_K" in BC.keys():
                K_0, E_K = BC["K_0"], BC["E_K"]  # non-instantaneous recomb
            value_BC = BoundaryConditionRecomb(phi, R_p, K_0, E_K, D_0, E_D, T)

        if BC["type"] not in FESTIM.helpers.bc_types["neumann"] and \
           BC["type"] not in FESTIM.helpers.bc_types["robin"] and \
           BC["type"] not in FESTIM.helpers.bc_types["dc"]:

            raise NameError("Unknown boundary condition type : " + BC["type"])
        if type_BC in FESTIM.helpers.bc_types["dc"]:
            if "component" in BC.keys():
                # Fetch the component of the BC
                component = BC["component"]
            else:
                # By default, component is solute (ie. 0)
                component = 0
            conservation_chemic_pot = False
            for mat in parameters["materials"]:
                if "S_0" in mat.keys():
                    conservation_chemic_pot = True
            if component == 0 and conservation_chemic_pot is True:
                # Store the non modified BC to be updated
                expressions.append(value_BC)
                # create modified BC based on solubility
                value_BC = BoundaryConditionTheta(
                    value_BC, volume_markers.mesh(), parameters["materials"],
                    volume_markers, T)
            expressions.append(value_BC)

            surfaces = BC['surfaces']
            if type(surfaces) is not list:
                surfaces = [surfaces]
            if V.num_sub_spaces() == 0:
                funspace = V
            else:  # if only one component, use subspace
                funspace = V.sub(component)
            for surface in surfaces:
                bci = DirichletBC(funspace, value_BC,
                                  surface_markers, surface)
                bcs.append(bci)

    return bcs, expressions
