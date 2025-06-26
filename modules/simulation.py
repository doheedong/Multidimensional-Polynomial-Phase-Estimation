import numpy as np
from .polynomial import Polynomial
from .estimator import compute_coefficient_proposed, compute_coefficient_dpt, compute_difference

def run_monte_carlo(algorithm_parameter, simulation_parameter):
    """
    Runs a Monte Carlo simulation for polynomial phase signal estimation.

    Args:
        algorithm_parameter (dict): Contains algorithm-specific parameters:
            - 'signal_basis' (SignalBasis): Instance defining the signal basis.
            - 'lag_set' (list[np.ndarray]): List of lag vectors.
            - 'averager_type' (str): Type of averager to use.
            - 'estimator_type' (str): Type of estimator ('do & lee & lozano' or 'DPT'), with default being 'do & lee & lozano'.
        simulation_parameter (dict): Contains simulation settings:
            - 'snrdb' (np.ndarray): Array of SNR values in dB.
            - 'n_trials' (int): Number of trials per SNR value.
            - 'stopping_condition' (str): Simulation stopping condition ('sample' or 'wrapping').
            - 'coefficient_type' (str): Type of coefficient to use ('random', 'zero', or 'custom').
            - 'coefficient_value' (np.ndarray, optional): Custom coefficient values if needed.

    Returns:
        If stopping_condition is 'sample':
            np.ndarray: Mean squared error (MSE) for each SNR value.
        If stopping_condition is 'wrapping':
            tuple: (mse, mse_wrapping, mse_no_wrapping, probability_wrapping), each as np.ndarray.
    """
    np.random.seed(0)

    signal_basis = algorithm_parameter['signal_basis']
    lag_set = algorithm_parameter['lag_set']
    estimator_type = algorithm_parameter.get('estimator_type', 'proposed')
    if estimator_type == 'proposed':
        compute_coefficient = compute_coefficient_proposed
    elif estimator_type == 'DPT':
        compute_coefficient = compute_coefficient_dpt
    else:
        raise ValueError(f"Error: Unsupported estimator type '{estimator_type}'. Use 'proposed' or 'DPT'.")

    snrdb = simulation_parameter['snrdb']
    n_trials = simulation_parameter['n_trials']
    stopping_condition = simulation_parameter['stopping_condition']
    coefficient_type = simulation_parameter['coefficient_type']
    coefficient_value = simulation_parameter.get('coefficient_value', None)

    snr = 10**(snrdb / 10)
    n_snr = len(snrdb)
    signal_size = tuple(signal_basis.signal_size)
    n_degree = len(signal_basis.degree_set)

    mse = np.zeros((n_snr,), dtype=float)
    mse_wrapping = np.zeros((n_snr,), dtype=float)
    mse_no_wrapping = np.zeros((n_snr,), dtype=float)
    count_wrapping = np.zeros((n_snr,), dtype=int)
    count_sample = np.zeros((n_snr,), dtype=int)
    if stopping_condition == 'sample':
        count_sample = np.zeros((n_snr,), dtype=int)
        while not np.all(count_sample >= n_trials):

            # Generate true coefficients
            if coefficient_type == 'random':
                true_coefficient = np.random.rand(n_degree) - 0.5
            elif coefficient_type == 'zero':
                true_coefficient = np.zeros(n_degree)
            elif coefficient_type == 'custom':
                if coefficient_value is None or len(coefficient_value) != n_degree:
                    raise ValueError("Custom coefficient_value must be provided and match n_degree.")
                true_coefficient = coefficient_value
            else:
                raise ValueError(f"Unknown coefficient_type: {coefficient_type}")

            # Generate true polynomial signal using Polynomial class
            poly = Polynomial(signal_basis, true_coefficient)
            polynomial_signal = poly.evaluate()
            polyphase_signal = np.exp(1j * 2 * np.pi * polynomial_signal)

            for i_snr in range(n_snr):
                if count_sample[i_snr] < n_trials:

                    noise_shape = signal_size
                    unit_noise = (1 / np.sqrt(2)) * (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape))
                    noisy_polyphase_signal = polyphase_signal + unit_noise / np.sqrt(snr[i_snr])

                    # Estimate coefficients using compute_coefficient and algorithm_parameter
                    estimated_coefficient = compute_coefficient(noisy_polyphase_signal, algorithm_parameter)

                    # Reconstruct polynomial signal using Polynomial class
                    reconstructed_poly = Polynomial(signal_basis, estimated_coefficient)
                    reconstructed_polynomial_signal = reconstructed_poly.evaluate()
                    reconstructed_polyphase_signal = np.exp(1j * 2 * np.pi * reconstructed_polynomial_signal)

                    mse_increment = np.sum(np.abs(reconstructed_polyphase_signal - polyphase_signal)**2)
                    mse[i_snr] += mse_increment
                    count_sample[i_snr] += 1

        mse = np.divide(mse, count_sample, out=np.zeros_like(mse), where=count_sample!=0)
        return mse

    elif stopping_condition == 'wrapping':
        while not np.all(count_wrapping >= n_trials):

            # Generate true coefficients
            if coefficient_type == 'random':
                true_coefficient = np.random.rand(n_degree) - 0.5
            elif coefficient_type == 'zero':
                true_coefficient = np.zeros(n_degree)
            elif coefficient_type == 'custom':
                if coefficient_value is None or len(coefficient_value) != n_degree:
                    raise ValueError("Custom coefficient_value must be provided and match n_degree.")
                true_coefficient = coefficient_value
            else:
                raise ValueError(f"Unknown coefficient_type: {coefficient_type}")

            # Generate true polynomial signal using Polynomial class
            poly = Polynomial(signal_basis, true_coefficient)
            polynomial_signal = poly.evaluate()
            polyphase_signal = np.exp(1j * 2 * np.pi * polynomial_signal)

            for i_snr in range(n_snr):
                if count_wrapping[i_snr] < n_trials:

                    noise_shape = signal_size
                    unit_noise = (1 / np.sqrt(2)) * (np.random.randn(*noise_shape) + 1j * np.random.randn(*noise_shape))
                    noisy_polyphase_signal = polyphase_signal + unit_noise / np.sqrt(snr[i_snr])

                    phase_noise = np.angle(1+unit_noise/np.sqrt(snr[i_snr]))
                    differenced_phase_noise = compute_difference(phase_noise, signal_basis.degree_set[-1], lag_set[-1], is_phase_difference=False)
                    is_wrapping = np.any(np.abs(differenced_phase_noise) > np.pi)

                    # Estimate coefficients using compute_coefficient and algorithm_parameter
                    estimated_coefficient = compute_coefficient_proposed(noisy_polyphase_signal, algorithm_parameter)

                    # Reconstruct polynomial signal using Polynomial class
                    reconstructed_poly = Polynomial(signal_basis, estimated_coefficient)
                    reconstructed_polynomial_signal = reconstructed_poly.evaluate()
                    reconstructed_polyphase_signal = np.exp(1j * 2 * np.pi * reconstructed_polynomial_signal)

                    mse_increment = np.sum(np.abs(reconstructed_polyphase_signal - polyphase_signal)**2)

                    if is_wrapping:
                        mse_wrapping[i_snr] += mse_increment
                        count_wrapping[i_snr] += 1
                    else:
                        mse_no_wrapping[i_snr] += mse_increment

                    mse[i_snr] += mse_increment
                    count_sample[i_snr] += 1
            
        non_wrapping_count = count_sample - count_wrapping
        mse = np.divide(mse, count_sample, out=np.zeros_like(mse), where=count_sample!=0)
        mse_wrapping = np.divide(mse_wrapping, count_wrapping, out=np.zeros_like(mse_wrapping), where=count_wrapping!=0)
        mse_no_wrapping = np.divide(mse_no_wrapping, non_wrapping_count, out=np.zeros_like(mse_no_wrapping), where=non_wrapping_count!=0)
        probability_wrapping = np.divide(count_wrapping, count_sample, out=np.zeros_like(count_wrapping, dtype=float), where=count_sample!=0)
        return mse, mse_wrapping, mse_no_wrapping, probability_wrapping
    else:
        raise ValueError(f"Invalid stopping condition: {stopping_condition}. Supported are 'sample' and 'wrapping'.")