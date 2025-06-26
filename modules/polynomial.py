import numpy as np

class Polynomial:
    """
    Represents a polynomial signal using a SignalBasis and coefficients.
    """

    def __init__(self, signal_basis, coefficient):
        """
        Initializes the Polynomial object.

        Args:
            signal_basis (SignalBasis): The SignalBasis object containing basis info.
            coefficient (np.ndarray): Array of polynomial coefficients.
        """
        self.signal_basis = signal_basis
        self.coefficient = coefficient

    def evaluate(self):
        """
        Computes the polynomial signal.

        Returns:
            np.ndarray: The polynomial signal.
        """
        # Use the basis set and signal size from the SignalBasis
        n_degree = len(self.signal_basis.degree_set)
        polynomial = np.zeros(self.signal_basis.signal_size, dtype=complex)

        for i_degree in range(n_degree):
            polynomial += self.coefficient[i_degree] * self.signal_basis.basis_set[i_degree]

        return polynomial