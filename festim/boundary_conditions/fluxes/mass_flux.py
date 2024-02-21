from festim import FluxBC, k_B
import fenics as f
import sympy as sp

def sieverts_constant(T, S_0, E_S):
    S = S_0 * f.exp(-E_S / k_B / T)
    return S

def henry_constant(T, H_0, E_H):
    H = H_0 * f.exp(-E_H / k_B / T)
    return H

class MassFlux(FluxBC):
    """
    FluxBC subclass for advective mass flux
    -D * grad(c) * n = h_mass * (c_interface_liquid - c_ext)

    Args:
        h_mass (float or sp.Expr): mass transfer coefficient (m/s)
        c_ext (float or sp.Expr): external concentration (1/m3)y
        solubility_law (string): the solubility law for the liquid - 'henry' or 'sievert'
        pre_exp_liquid (float): pre-exponential factor for the liquid solubility law (m-3/Pa0.5 or m-3/Pa)
        activation_energy_liquid (float): activation energy for the liquid solubility law (eV)
        S_0 (float): Sievert's constant pre-exponential factor (m-3/Pa0.5)
        E_S (float): Sievert's constant activation energy (eV)
        surfaces (list or int): the surfaces of the BC

    Reference: Bergman, T. L., Bergman, T. L., Incropera, F. P., Dewitt, D. P.,
    & Lavine, A. S. (2011). Fundamentals of heat and mass transfer. John Wiley & Sons.
    """

    def __init__(self, h_coeff, c_ext, solubility_law, pre_exp_liquid, activation_energy_liquid, S_0, E_S, surfaces) -> None:
        self.h_coeff = h_coeff
        self.c_ext = c_ext
        self.pre_exp_liquid = pre_exp_liquid
        self.activation_energy_liquid = activation_energy_liquid
        self.S_0 = S_0
        self.E_S = E_S
        self.solubility_law = solubility_law        
        super().__init__(surfaces=surfaces, field=0)

    def create_form(self, T, solute):
        h_coeff = f.Expression(sp.printing.ccode(self.h_coeff), t=0, degree=1)
        c_ext = f.Expression(sp.printing.ccode(self.c_ext), t=0, degree=1)
        S = sieverts_constant(T, self.S_0, self.E_S)

        if self.solubility_law == "sievert":
            solubility_liquid = sieverts_constant(T, self.pre_exp_liquid, self.activation_energy_liquid)
            c_interface_liquid = (solute/S) * solubility_liquid
        elif self.solubility_law == "henry":
            solubility_liquid = henry_constant(T, self.pre_exp_liquid, self.activation_energy_liquid)
            c_interface_liquid = (solute / S)**2 * solubility_liquid
        else:
            raise ValueError("Invalid solubility law. Choose between 'sievert' or 'henry'")

        self.form = -h_coeff * (c_interface_liquid - c_ext)
        self.sub_expressions = [h_coeff, c_ext]
