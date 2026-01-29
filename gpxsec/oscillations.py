import nuTens as nt
import torch
import math as m

import typing


class OscillationPropagator(torch.nn.Module):
    """
    Module for calculating oscillation probabilities

    """

    def __init__(self):
        """
        Docstring for __init__
        
        """

        super().__init__()

        ## set initial mass squared diffs
        self.dmsq12  = torch.nn.Parameter(torch.randn(1) * 0.0001 + 0.0025 )
        self.dmsq13  = torch.nn.Parameter(torch.randn(1) * 0.0001 + 0.0025 )

        ## will be set depending on the type of propagator
        self.deltaCP = None
        self.theta12 = None
        self.theta13 = None
        self.theta23 = None

        ## needed if we are using default propagator
        self.pmns = None

        ## set up the propagator
        self.propagator = None

        self._setup_propagator()

    def _setup_propagator(self) -> None:

        self.deltaCP = torch.nn.Parameter(torch.randn(1) * 0.01 + m.pi / 4.0)
        self.theta12 = torch.nn.Parameter(torch.randn(1) * 0.01 + m.pi / 3.0)
        self.theta13 = torch.nn.Parameter(torch.randn(1) * 0.01 + m.pi / 3.0)
        self.theta23 = torch.nn.Parameter(torch.randn(1) * 0.01 + m.pi / 3.0)

        ## make the matrix
        self.propagator = nt.propagator.DPpropagator(295.0 * nt.units.km, False, 2.79, 10)
        self.propagator.set_parameters(
            nt.tensor.Tensor.from_torch_tensor(self.theta12), 
            nt.tensor.Tensor.from_torch_tensor(self.theta23), 
            nt.tensor.Tensor.from_torch_tensor(self.theta13), 
            nt.tensor.Tensor.from_torch_tensor(self.deltaCP), 
            nt.tensor.Tensor.from_torch_tensor(self.dmsq12), 
            nt.tensor.Tensor.from_torch_tensor(self.dmsq13)
        )

    def set_parameter_values(self, parameter_dict: typing.Dict[str, float]) -> None:
        """
        Set values of oscillation parameters
        
        :param parameter_dict: dictionary of parameter names and values
        should look something like {'deltaCP': m.pi / 2, 'theta12': 0.3, 'theta13': 0.2, 'theta23': 0.6, 'dmsq12': 0.00025, 'dmsq13':0.0025}
        :type parameter_dict: typing.Dict[str, float]
        :raises AttributeError: If the module does not have a parameter with the specified name
        """

        for name, value in parameter_dict.items():

            self.set_parameter_value(name, value)
            
    def set_parameter_value(self, parameter_name: str, value: float) -> None:
        """
        Set value of a single oscillation parameter
        
        :param parameter_name: name of the parameter
        :type parameter_name: str
        :param value: The new value
        :type value: float
        :raises AttributeError: If the module does not have a parameter with the specified name
        """

        with torch.no_grad():
            if hasattr(self, parameter_name):
                getattr(self, parameter_name)[0] = value

            else:
                raise AttributeError(f"OscillationPropagator does not have attribute {parameter_name}")
            

    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        """
        Docstring for forward
        
        :param energies: Energies to calculate oscillation probs for
        :type energies: torch.Tensor
        :return: Oscillation probabilities for the given energies
        :rtype: torch.Tensor
        """

        ## need to convert energies to complex... should fix this in nuTens
        complex_energies = energies.to(torch.complex64)
        self.propagator.set_energies(nt.tensor.Tensor.from_torch_tensor(complex_energies))

        probs = self.propagator.calculate_probabilities()

        return probs
    

class OscillationPropagatorGeneral(OscillationPropagator):

    def __init__(self):
        """
        Docstring for __init__
        """

        super().__init__()

    def _setup_propagator(self) -> None:

        self.pmns = nt.propagator.PMNSmatrix()

        self.deltaCP = torch.nn.Parameter(self.pmns.get_delta_cp_tensor().torch_tensor())
        self.theta12 = torch.nn.Parameter(self.pmns.get_theta_12_tensor().torch_tensor())
        self.theta13 = torch.nn.Parameter(self.pmns.get_theta_13_tensor().torch_tensor())
        self.theta23 = torch.nn.Parameter(self.pmns.get_theta_23_tensor().torch_tensor())

        ## set up the propagator object
        self.propagator = nt.propagator.Propagator(3, 295.0 * nt.units.km)
        self.matter_solver = nt.propagator.ConstDensitySolver(3, 2.79)

        self.propagator.set_matter_solver(self.matter_solver)
    
    def forward(self, energies: torch.Tensor) -> torch.Tensor:

        # build tensor of masses
        masses = torch.stack(
            [
                torch.zeros(1),
                self.dmsq12.sqrt(),
                self.dmsq13.sqrt()
            ],
            dim = 1
        )

        self.propagator.set_mixing_matrix(self.pmns.build())
        self.propagator.set_masses(nt.tensor.Tensor.from_torch_tensor(masses))

        ## need to convert energies to complex... should fix this in nuTens
        complex_energies = energies.to(torch.complex64)
        self.propagator.set_energies(nt.tensor.Tensor.from_torch_tensor(complex_energies))

        probs = self.propagator.calculate_probabilities()

        return probs
    