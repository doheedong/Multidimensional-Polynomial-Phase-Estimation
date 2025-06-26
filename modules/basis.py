import numpy as np
import math

def compute_binomial(n, k):
    """
    Computes the generalized binomial coefficient (n choose k) = n * (n-1) * ... * (n-k+1) / k!.

    Args:
        n (np.ndarray): The 'n' value in C(n, k). Can be a scalar or a NumPy array.
        k (int): The 'k' value in C(n, k). If negative, the result is zero.

    Returns:
        np.ndarray: The binomial coefficient, with the same shape as 'n' and dtype int.
    """
    if k < 0:
        return np.zeros_like(n, dtype=int)
    else:
        output = np.ones_like(n, dtype=int)
        for i in range(k):
            output = output * (n - i)
        output = output // math.factorial(k)  # Use integer division
        return output

def compute_monomial(n, k):
    """
    Computes n^k / k!.

    Args:
        n (np.ndarray): The base 'n' in n^k. Can be a scalar or a NumPy array.
        k (int): The exponent 'k'. If negative, the result is zero.

    Returns:
        np.ndarray: The monomial term, with the same shape as 'n'.
    """
    if k < 0:
        return np.zeros_like(n, dtype=float)
    else:
        return (n**k) / math.factorial(k)

def generate_degree_set(n_dimension, max_degree):
    """
    Generates a set of multi-index (multi-dimensional) degree vectors for multi-dimensional polynomials.

    It generates all possible non-negative integer vectors [d_1, d_2, ..., d_{n_dimension}] (multi-indices) such that sum(d_i) <= max_degree.

    Args:
        n_dimension (int): The number of dimensions.
        max_degree (int): The maximum total degree.

    Returns:
        list[np.ndarray]: A list where each element is a 1D numpy array of length `n_dimension`, representing a multi-index degree vector for a polynomial term.

    Examples:
        n_dimension = 1, max_degree = 2
            [[0], [1], [2]]
        n_dimension = 2, max_degree = 2
            [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]]
    """
    degree_set = []
    def compositions(n, k):
        """Generate all k-tuples of non-negative integers summing to n."""
        if k == 1:
            yield (n,)
        else:
            for i in range(n + 1):
                for tail in compositions(n - i, k - 1):
                    yield (i,) + tail

    for total_degree in range(max_degree + 1):
        for deg_tuple in compositions(total_degree, n_dimension):
            degree_set.append(np.array(deg_tuple, dtype=int))
    return degree_set

def generate_basis_set(signal_size, degree_set, basis_type):
    """
    Generates a set of polynomial basis functions based on specified settings.

    This function creates basis functions (e.g., binomial, monomial) for a given
    signal size and degree set across multiple dimensions.

    Args:
        signal_size (np.ndarray): Dimensions of the signal.
        degree_set (list[np.ndarray]): A list of degree vectors for each term.
        basis_type (str): Type of basis to generate ('binomial' or 'monomial').

    Returns:
        list[np.ndarray]: A list where each element is a numpy array representing
                          a basis function for the corresponding degree in degree_set.

    Raises:
        ValueError: If an unsupported basis type is provided.
    """
    if basis_type == 'binomial':
        compute_term = compute_binomial
    elif basis_type == 'monomial':
        compute_term = compute_monomial
    else:
        raise ValueError(f"Error: Unsupported basis type '{basis_type}'. Use 'binomial' or 'monomial'.")

    n_degree = len(degree_set)
    n_dimension = len(signal_size)

    basis_set = [None] * n_degree

    for i_degree in range(n_degree):
        degree = degree_set[i_degree]
        basis = np.ones(signal_size, dtype=float)
        for i_dimension in range(n_dimension):
            basis1d = compute_term(np.arange(signal_size[i_dimension]), degree[i_dimension])
            reshape_dims = [1] * n_dimension
            reshape_dims[i_dimension] = signal_size[i_dimension]
            basis1d_reshaped = basis1d.reshape(tuple(reshape_dims))
            basis = basis * basis1d_reshaped
        basis_set[i_degree] = basis
    return basis_set

def compute_extended_degree_set(degree_set):
    """
    Computes the extended degree set for a given set of polynomial degrees.

    The extended degree set includes all combinations of the original degree vectors
    raised to powers from 0 to the maximum degree in the degree set.

    Args:
        degree_set (list[np.ndarray]): The original set of degree vectors.

    Returns:
        tuple: (extended_degree_set, is_contained)
            extended_degree_set (list[np.ndarray]): A list of extended degree vectors, where each vector is a NumPy array.
            is_contained (list[bool]): A list of boolean flags, where each flag indicates whether the corresponding element of degree_set is in extended_degree_set.
    """

    extended_degree_set = set()
    for deg in degree_set:
        for idx in np.ndindex(*[d + 1 for d in deg]):
            extended_degree_set.add(tuple(idx))
    extended_degree_set = [np.array(idx, dtype=int) for idx in sorted(extended_degree_set)]

    # Compute boolean flags
    is_contained = [any(np.array_equal(deg2, deg1) for deg2 in degree_set) for deg1 in extended_degree_set]

    return extended_degree_set, is_contained

class SignalBasis:
    def __init__(self, signal_size, degree_set, basis_set):
        """
        Initializes the SignalBasis object, which is 'n_dimension'D.

        Args:
            signal_size (np.ndarray): The size or shape of the signal. It is a 1D NumPy array of size 'n_dimension'.
            degree_set (list[np.ndarray]): The set of degree vectors for the polynomial basis. Each degree vector is a 1D NumPy array of size 'n_dimension'.
            basis_set (list[np.ndarray]): The list of basis functions (NumPy arrays) corresponding to each degree. 
                  Each basis is 'n_dimension'D NumPy array whose shape matches signal_size. 
        """

        self.signal_size = signal_size
        self.degree_set = degree_set
        self.basis_set = basis_set

        # Dimensionality checks 
        n_dimension = len(self.signal_size)
        n_degree = len(self.degree_set)
        if len(self.basis_set) != n_degree:
            raise ValueError(f"basis_set length ({len(self.basis_set)}) does not match degree_set length ({n_degree})")
        for i, deg in enumerate(self.degree_set):
            if len(deg) != n_dimension:
                raise ValueError(f"degree_set[{i}] has length {len(deg)}, expected {n_dimension}")
        for i, basis in enumerate(self.basis_set):
            if basis.shape != tuple(self.signal_size):
                raise ValueError(f"basis_set[{i}] has shape {basis.shape}, expected {tuple(self.signal_size)}")
    
    def copy(self):
        """
        Creates a copy of the SignalBasis object.

        Returns:
            SignalBasis: A new instance of SignalBasis with the same attributes.
        """
        return SignalBasis(self.signal_size.copy(), 
                           [deg.copy() for deg in self.degree_set], 
                           [basis.copy() for basis in self.basis_set])
        
    def compute_fisher_matrix(self) -> np.ndarray:
        """
        Computes the Fisher Information Matrix for the basis functions.

        Returns:
            np.ndarray: The Fisher Information Matrix, a square matrix with dimensions 'n_degree' x 'n_degree', where 'n_degree' is the number of basis functions.
        """
        # Validate that all basis arrays have the same shape
        basis_shapes = [b.shape for b in self.basis_set]
        if not all(s == basis_shapes[0] for s in basis_shapes):
            raise ValueError(f"All basis arrays in basis_set must have the same shape, but got shapes: {basis_shapes}")

        # Stack basis vectors as flattened arrays (each as a row)
        basis_matrix = np.stack([b.flatten() for b in self.basis_set], axis=0)
        fisher_matrix = basis_matrix @ basis_matrix.T
        return fisher_matrix