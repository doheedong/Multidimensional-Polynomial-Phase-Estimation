import numpy as np
from .basis import compute_binomial, compute_extended_degree_set, generate_basis_set

def compute_weight_1d(signal_size, degree, lag=1):
    """
    Computes 1D weights.

    This function implements the unnormalized weighting logic and then normalizes
    the resulting weights such that their sum is 1.

    Args:
        signal_size (int): The size of the 1D signal.
        degree (int): The degree of the polynomial.
        lag (int, optional): The lag value. Defaults to 1.

    Returns:
        np.ndarray: The normalized 1D weight array.
    """
    output_len = signal_size - degree * lag
    if output_len < 1:
        raise ValueError('Error: the lag value is too large, resulting in signal length < 1.')

    if lag == 1:
        n = np.arange(signal_size - degree)
        term1 = compute_binomial(n + degree, degree).astype(np.int64)
        term2 = compute_binomial(signal_size - n - 1, degree).astype(np.int64)
        # For very large signal sizes, weight may overflow; though, to produce the results in the paper, int64 suffices.
        weight = term1 * term2
    else:
        weight = np.zeros(output_len, dtype=float)
        for stride_start in range(lag):
            stride_indices = np.arange(stride_start, output_len, lag)
            effective_signal_size = len(stride_indices) + degree
            stride_weight = compute_weight_1d(effective_signal_size, degree, 1)
            weight[stride_indices] = stride_weight

    sum_weight = np.sum(weight)
    weight = weight / sum_weight
    return weight

def compute_weight(signal_size, degree, lag):
    """
    Computes multi-dimensional weights.

    All inputs are numpy 1D arrays of size n_dimension.

    Args:
        signal_size (np.ndarray): 1D array of signal dimensions.
        degree (np.ndarray): 1D array of degrees for each dimension.
        lag (np.ndarray): 1D array of lag values for each dimension.

    Returns:
        np.ndarray: The computed multi-dimensional weight array.

    Raises:
        ValueError: If signal dimensions, degree, or lag values are incompatible.
    """
    signal_size = np.array(signal_size).flatten()
    degree = np.array(degree).flatten()
    lag = np.array(lag).flatten()

    n_dimension = signal_size.size
    if not (degree.size == lag.size == n_dimension):
        raise ValueError("All inputs must be 1D numpy arrays of the same size (n_dimension).")

    if n_dimension == 0:
        # Scalar case
        return np.ones(1, dtype=float)

    output_shape_list = [int(signal_size[i] - degree[i] * lag[i]) for i in range(n_dimension)]
    if any(s < 1 for s in output_shape_list):
        raise ValueError("Invalid signal dimensions or lag/degree values leading to non-positive output shape.")
    output_shape = tuple(output_shape_list)

    weight = np.ones(output_shape, dtype=float)
    for i_dimension in range(n_dimension):
        weight1d = compute_weight_1d(signal_size[i_dimension], degree[i_dimension], lag[i_dimension])
        reshape_dims = [1] * n_dimension
        reshape_dims[i_dimension] = output_shape[i_dimension]
        weight1d_reshaped = weight1d.reshape(tuple(reshape_dims))
        weight1d_reshaped = weight1d.reshape(reshape_dims)
        weight = weight * weight1d_reshaped

    return weight

def compute_difference(signal, n_difference, lag=None, is_phase_difference=False):
    """
    Computes the difference or phase difference of a signal along specified dimensions.

    Args:
        signal (np.ndarray): The input signal.
        n_difference (np.ndarray): Number of differences per dimension.
        lag (np.ndarray, optional): Lag per dimension. Must be a numpy array of positive integers. Defaults to 1 for each dimension.
        is_phase_difference (bool): If True, computes phase difference (complex-valued); if False, computes real difference.

    Returns:
        np.ndarray: The transformed signal after computing differences.
    """
    n_dimension = signal.ndim
    if lag is None:
        lag = np.ones(n_dimension, dtype=int)
    else:
        lag = np.array(lag, dtype=int)
        if not np.all((lag > 0) & (lag == np.round(lag))):
            raise ValueError("lag must be a numpy array of positive integers.")

    n_difference = np.array(n_difference, dtype=int)
    signal_size = np.array(signal.shape, dtype=int)
    output_shape_list = [int(signal_size[i] - n_difference[i] * lag[i]) for i in range(n_dimension)]
    if any(s < 1 for s in output_shape_list):
        raise ValueError("Invalid signal dimensions or lag/degree values leading to non-positive output shape.")

    current_signal = signal.copy()

    for i_dimension in range(n_dimension):
        for _ in range(n_difference[i_dimension]):
            dim_len = current_signal.shape[i_dimension]
            current_lag = lag[i_dimension]

            idx1 = [slice(None)] * current_signal.ndim
            idx2 = [slice(None)] * current_signal.ndim
            idx1[i_dimension] = slice(current_lag, dim_len)
            idx2[i_dimension] = slice(0, dim_len - current_lag)

            if is_phase_difference:
                current_signal = current_signal[tuple(idx1)] * np.conj(current_signal[tuple(idx2)])
            else:
                current_signal = current_signal[tuple(idx1)] - current_signal[tuple(idx2)]

    return current_signal

def compute_average(signal, weight, averager_type):
    """
    Computes the weighted average of a signal based on the specified averager type.

    Args:
        signal (np.ndarray): The input complex signal.
        weight (np.ndarray): Weighting array, whose size must match with signal.
        averager_type (str): Type of averager ('linear', 'kay', 'lovell & williamson', 'do & lee & lozano').

    Returns:
        float: The computed average value.

    Raises:
        ValueError: If an unsupported averager type is provided.
    """
    circular_mean = np.sum(signal)
    if averager_type == 'linear':
        average = np.sum(weight * np.angle(signal)) / (2 * np.pi)
    elif averager_type == 'kay':
        average = np.angle(np.sum(weight * signal)) / (2 * np.pi)
    elif averager_type == 'lovell & williamson':
        average = np.angle(np.sum(weight * signal / np.abs(signal))) / (2 * np.pi)
    elif averager_type == 'do & lee & lozano':
        average = np.angle(circular_mean * np.exp(1j * np.sum(weight * np.angle(signal * np.conj(circular_mean))))) / (2 * np.pi)
    else:
        raise ValueError(f"Invalid averager type: {averager_type}. Supported types are 'linear', 'kay', 'lovell & williamson', 'do & lee & lozano'.")
    return average

def compute_coefficient_proposed(noisy_polyphase_signal, algorithm_parameter):
    """
    Estimates polynomial phase coefficients from a noisy polyphase signal.

    Args:
        noisy_polyphase_signal (np.ndarray): The input noisy complex polyphase signal.
        algorithm_parameter (dict): Dictionary containing:
            - 'signal_basis' (SignalBasis): an instance of SignalBasis
            - 'lag_set' (list[np.ndarray]): list of lag vectors, each a 1D numpy array.
            - 'averager_type' (str): averager type ('linear', 'kay', etc.)
            - 'is_naive' (bool): whether to use naive estimation or extended degree set with Fisher information matrix.

    Returns:
        np.ndarray: An array of estimated polynomial coefficients.
    """
    algorithm_parameter = algorithm_parameter.copy()
    signal_basis = algorithm_parameter['signal_basis']
    lag_set = algorithm_parameter['lag_set']
    averager_type = algorithm_parameter['averager_type']
    is_naive = algorithm_parameter['is_naive']

    degree_set = signal_basis.degree_set
    basis_set = signal_basis.basis_set
    signal_size = signal_basis.signal_size
    n_degree = len(degree_set)
    n_lag = len(lag_set)

    (extended_degree_set, is_contained) = compute_extended_degree_set(degree_set)
    if all(is_contained) or is_naive:
        estimated_coefficient = np.zeros(n_degree, dtype=float)
        residual_signal = noisy_polyphase_signal.copy()
        for i_degree in range(n_degree - 1, -1, -1):
            degree = degree_set[i_degree]
            basis = basis_set[i_degree]
            for i_lag in range(n_lag):
                lag = np.array(lag_set[i_lag])
                weight = compute_weight(signal_size, degree, lag)
                transformed_signal = compute_difference(residual_signal, degree, lag, is_phase_difference=True)
                lag_degree_prod = np.prod(lag ** np.array(degree))
                coefficient_increment = compute_average(transformed_signal, weight, averager_type) / lag_degree_prod

                estimated_coefficient[i_degree] += coefficient_increment
                residual_signal = residual_signal * np.exp(-1j * 2 * np.pi * coefficient_increment * basis)
    else:
        extended_basis_set = generate_basis_set(signal_size, extended_degree_set, basis_type='binomial')
        i = 0
        for i_extended, contained in enumerate(is_contained):
            if contained:
                extended_basis_set[i_extended] = basis_set[i]
                i += 1
        extended_signal_basis = signal_basis.copy()
        extended_signal_basis.degree_set = extended_degree_set
        extended_signal_basis.basis_set = extended_basis_set

        algorithm_parameter['signal_basis'] = extended_signal_basis
        fisher_matrix = extended_signal_basis.compute_fisher_matrix()
        estimated_extended_coefficient = compute_coefficient_proposed(noisy_polyphase_signal, algorithm_parameter)
        contained_indices = np.where(is_contained)[0]
        estimated_coefficient = (
            np.linalg.inv(fisher_matrix[np.ix_(contained_indices, contained_indices)])
            @ fisher_matrix[contained_indices, :] @ estimated_extended_coefficient
        )
    return estimated_coefficient

def compute_coefficient_dpt(noisy_polyphase_signal, algorithm_parameter):
    """
    Estimates polynomial phase coefficients from a noisy polyphase signal using discrete polynomial phase transform in [R1] and [R2].
        [R1] S. Peleg and B. Friedlander, "The discrete polynomial-phase transform," in IEEE Transactions on Signal Processing, 1995.
        [R2] B. Friedlander and J. M. Francos, "Model based phase unwrapping of 2-D signals," in IEEE Transactions on Signal Processing, 1996.
    Different from the references, fine search is performed using gradient descent with backtracking line search.
    Only supports 1D and 2D signals with the set of degrees being
        [[0], [1], ..., [max_degree]] for 1D
        [[0, 0], ..., [0, max_degree], [1, 0], ..., [1, max_degree-1], ... , [2, 0], ... , [2, max_degree-2], ... , [max_degree, 0]] for 2D.
    
    Args:
        noisy_polyphase_signal (np.ndarray): The input noisy complex polyphase signal.
        algorithm_parameter (dict): Dictionary containing:
            - 'signal_basis' (SignalBasis): an instance of SignalBasis
            - 'lag_set' (list[np.ndarray]): list of lag vectors, each a 1D numpy array.

    Returns:
        np.ndarray: An array of estimated polynomial coefficients.
    """
    # Parameters for coarse/fine search (specifically, alpha and beta are parameters for backtracking line search)
    zero_padding_factor = 4
    step_size = 1e-5
    alpha = 0.5
    beta = 0.5
    threshold = 1e-5

    algorithm_parameter = algorithm_parameter.copy()
    signal_basis = algorithm_parameter['signal_basis']
    lag_set = algorithm_parameter['lag_set']

    degree_set = signal_basis.degree_set
    basis_set = signal_basis.basis_set
    signal_size = np.array(signal_basis.signal_size)
    n_degree = len(degree_set)
    n_dimension = signal_size.size
    n_layer = max([sum(degree) for degree in degree_set]) + 1

    if n_dimension == 1:
        if not n_degree == n_layer:
            raise ValueError("Only supports contiguous set of degrees")
    elif n_dimension == 2:
        if not n_degree == int(n_layer * (n_layer + 1) / 2):
            raise ValueError("Only supports triangular set of degrees")
    else:
        raise ValueError("Only supports 1D and 2D signals.")
    if not len(lag_set) == 1:
        raise ValueError("Only supports a single lag vector.")

    lag = np.array(lag_set[0])
    coefficient = np.zeros(n_degree)
    residual_signal = noisy_polyphase_signal.copy()

    if n_dimension == 1:
        for i_degree in range(n_layer - 2, -1, -1):
            degree = degree_set[i_degree]
            transformed_signal = compute_difference(residual_signal, degree, lag, is_phase_difference=True)
            def demodulate(est):
                return transformed_signal * np.exp(-1j * 2 * np.pi * est * idx)
            def grad(est):
                demod = demodulate(est)
                return 4 * np.pi * np.imag(np.sum(idx * demod) * np.conj(np.sum(demod))) / signal_size[0]
            def obj(est):
                return np.abs(np.sum(demodulate(est))) ** 2 / signal_size[0]

            transformed_signal_len = signal_size[0] - lag[0] * degree[0]
            padded_shape = (zero_padding_factor * signal_size[0],)
            padded_transformed_signal = np.zeros(padded_shape, dtype=complex)
            padded_transformed_signal[:transformed_signal_len] = transformed_signal
            periodogram = np.abs(np.fft.fftn(padded_transformed_signal))
            i_max = np.argmax(periodogram)
            estimate = (i_max) / (signal_size[0] * zero_padding_factor)
            estimate -= round(estimate)
            idx = np.arange(transformed_signal_len)

            gradient = grad(estimate)
            while True:
                step_size_next = step_size
                while True:
                    estimate_next = estimate + step_size_next * gradient
                    if obj(estimate_next) >= obj(estimate) + alpha * step_size_next * gradient ** 2:
                        break
                    step_size_next *= beta
                estimate += step_size_next * gradient
                gradient = grad(estimate)
                if step_size_next * abs(gradient) < threshold:
                    break
            coefficient[i_degree + 1] = estimate / (lag[0] ** degree[0])
            residual_signal = residual_signal * np.exp(-1j * 2 * np.pi * coefficient[i_degree + 1] * basis_set[i_degree + 1])
    elif n_dimension == 2:
        i_degree = int((n_layer - 1) * n_layer / 2) - 1
        for i_layer in range(n_layer - 2, -1, -1):
            signal_layer = np.ones_like(noisy_polyphase_signal)
            for _ in range(i_layer, -1, -1):
                degree = degree_set[i_degree]
                transformed_signal = compute_difference(residual_signal, degree, lag, is_phase_difference=True)
                def demodulate(est):
                    return transformed_signal * np.exp(-1j * 2 * np.pi * (est[0] * grid1 + est[1] * grid2))
                def grad(est):
                    demod = demodulate(est)
                    g = np.zeros(2)
                    norm = signal_size[0] * signal_size[1]
                    g[0] = 4 * np.pi * np.imag(np.sum(grid1 * demod) * np.conj(np.sum(demod))) / norm
                    g[1] = 4 * np.pi * np.imag(np.sum(grid2 * demod) * np.conj(np.sum(demod))) / norm
                    return g
                def obj(est):
                    return np.abs(np.sum(demodulate(est))) ** 2 / (signal_size[0] * signal_size[1])

                s1 = signal_size[0] - lag[0] * degree[0]
                s2 = signal_size[1] - lag[1] * degree[1]
                padded_shape = (zero_padding_factor * signal_size[0], zero_padding_factor * signal_size[1])
                padded_transformed_signal = np.zeros(padded_shape, dtype=complex)
                padded_transformed_signal[:s1, :s2] = transformed_signal
                periodogram = np.abs(np.fft.fftn(padded_transformed_signal))
                i_max = np.argmax(periodogram)
                i_max1, i_max2 = np.unravel_index(i_max, periodogram.shape)
                estimate = np.zeros(2)
                estimate[0] = i_max1 / (signal_size[0] * zero_padding_factor)
                estimate[0] -= round(estimate[0])
                estimate[1] = i_max2 / (signal_size[1] * zero_padding_factor)
                estimate[1] -= round(estimate[1])
                idx1 = np.arange(s1)
                idx2 = np.arange(s2)
                grid1, grid2 = np.meshgrid(idx1, idx2, indexing='ij')
                gradient = grad(estimate)
                while True:
                    step_size_next = step_size
                    while True:
                        estimate_next = estimate + step_size_next * gradient
                        if obj(estimate_next) >= obj(estimate) + alpha * step_size_next * np.linalg.norm(gradient) ** 2:
                            break
                        step_size_next *= beta
                    estimate += step_size_next * gradient
                    gradient = grad(estimate)
                    if step_size_next * np.linalg.norm(gradient) < threshold:
                        break
                # Assign coefficients for each dimension
                coefficient[i_degree + i_layer + 2] = estimate[0] / (lag[0] ** degree[0] * lag[1] ** degree[1])
                coefficient[i_degree + i_layer + 1] = estimate[1] / (lag[0] ** degree[0] * lag[1] ** degree[1])
                signal_layer = signal_layer * np.exp(1j * 2 * np.pi * coefficient[i_degree + i_layer + 2] * basis_set[i_degree + i_layer + 2])
                i_degree -= 1
            signal_layer = signal_layer * np.exp(1j * 2 * np.pi * coefficient[i_degree + i_layer + 2] * basis_set[i_degree + i_layer + 2])
            residual_signal = residual_signal * np.conj(signal_layer)
        
    # Estimate the constant term
    coefficient[0] = np.mean(np.angle(residual_signal)) / (2 * np.pi)
    return coefficient

def compute_cramer_rao_bound(signal_basis, snrdb):
    """
    Computes the reconstruction MSE under CRB attainment.

    Args:
        signal_basis (SignalBasis): An instance of SignalBasis containing basis info.
        snrdb (np.ndarray): Signal-to-Noise Ratio in dB.

    Returns:
        np.ndarray: The reconstruction MSE for each SNR value.
    """
    n_degree = len(signal_basis.degree_set)
    
    snr = 10**(snrdb / 10)
    mse_crb = n_degree/2/snr

    return mse_crb