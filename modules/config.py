import numpy as np
from .basis import generate_degree_set, generate_basis_set, SignalBasis

def get_parameter(i_figure, i_curve=0):
    """
    Retrieves simulation and algorithm parameters for a given figure identifier.

    Args:
        i_figure (str): Figure identifier (e.g., '5', '10a').
        i_curve (int, optional): Curve index for specific settings.

    Returns:
        algorithm_parameter (dict): Contains:
            - 'signal_basis' (SignalBasis)
            - 'lag_set' (list[np.ndarray])
            - 'averager_type' (str)
        simulation_parameter (dict): Contains:
            - 'snrdb' (np.ndarray)
            - 'n_trials' (int)
            - 'coefficient_type' (str)
            - 'coefficient_value' (np.ndarray or None)
            - 'stopping_condition' (str)
    """

    settings = {
        '5': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([64])]),
        '6': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([64])]),
        '9a': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([64])]),
        '9b': dict(n_dimension=1, max_degree=2, signal_size_set=[np.array([64])]),
        '9c': dict(n_dimension=1, max_degree=3, signal_size_set=[np.array([64])]),
        '10a': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([16]), np.array([64]), np.array([256]), np.array([1024])]),
        '10b': dict(n_dimension=1, max_degree=2, signal_size_set=[np.array([16]), np.array([64]), np.array([256]), np.array([1024])]),
        '10c': dict(n_dimension=1, max_degree=3, signal_size_set=[np.array([16]), np.array([64]), np.array([256]), np.array([1024])]),
        '10d': dict(n_dimension=2, max_degree=1, signal_size_set=[np.array([4, 4]), np.array([8, 8]), np.array([16, 16]), np.array([32, 32])]),
        '10e': dict(n_dimension=2, max_degree=2, signal_size_set=[np.array([4, 4]), np.array([8, 8]), np.array([16, 16]), np.array([32, 32])]),
        '10f': dict(n_dimension=2, max_degree=3, signal_size_set=[np.array([4, 4]), np.array([8, 8]), np.array([16, 16]), np.array([32, 32])]),
        '10g': dict(n_dimension=3, max_degree=1, signal_size_set=[np.array([4, 4, 4]), np.array([8, 8, 8]), np.array([16, 16, 16]), np.array([32, 32, 32])]),
        '10h': dict(n_dimension=3, max_degree=2, signal_size_set=[np.array([4, 4, 4]), np.array([8, 8, 8]), np.array([16, 16, 16]), np.array([32, 32, 32])]),
        '10i': dict(n_dimension=3, max_degree=3, signal_size_set=[np.array([4, 4, 4]), np.array([8, 8, 8]), np.array([16, 16, 16]), np.array([32, 32, 32])]),
        '10j': dict(n_dimension=4, max_degree=1, signal_size_set=[np.array([4, 4, 4, 4]), np.array([8, 8, 8, 8]), np.array([16, 16, 16, 16]), np.array([32, 32, 32, 32])]),
        '10k': dict(n_dimension=4, max_degree=2, signal_size_set=[np.array([4, 4, 4, 4]), np.array([8, 8, 8, 8]), np.array([16, 16, 16, 16]), np.array([32, 32, 32, 32])]),
        '10l': dict(n_dimension=4, max_degree=3, signal_size_set=[np.array([4, 4, 4, 4]), np.array([8, 8, 8, 8]), np.array([16, 16, 16, 16]), np.array([32, 32, 32, 32])]),
        '11': dict(n_dimension=1, max_degree=3, signal_size_set=[np.array([16]), np.array([64]), np.array([256]), np.array([1024])]),
        '12a': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([64])]),
        '12b': dict(n_dimension=1, max_degree=2, signal_size_set=[np.array([64])]),
        '12c': dict(n_dimension=1, max_degree=3, signal_size_set=[np.array([64])]),
        '13a': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([64])]),
        '13b': dict(n_dimension=1, max_degree=2, signal_size_set=[np.array([64])]),
        '13c': dict(n_dimension=1, max_degree=3, signal_size_set=[np.array([64])]),
        '14a': dict(n_dimension=2, max_degree=1, signal_size_set=[np.array([32, 32])]),
        '14b': dict(n_dimension=3, max_degree=1, signal_size_set=[np.array([32, 32, 32])]),
        '14c': dict(n_dimension=4, max_degree=1, signal_size_set=[np.array([32, 32, 32, 32])]),
        '15a': dict(n_dimension=1, max_degree=1, signal_size_set=[np.array([64])]),
        '15b': dict(n_dimension=1, max_degree=2, signal_size_set=[np.array([64])]),
        '15c': dict(n_dimension=1, max_degree=3, signal_size_set=[np.array([64])]),
        '15d': dict(n_dimension=2, max_degree=1, signal_size_set=[np.array([16, 16])]),
        '15e': dict(n_dimension=2, max_degree=2, signal_size_set=[np.array([16, 16])]),
        '15f': dict(n_dimension=2, max_degree=3, signal_size_set=[np.array([16, 16])]),
    }

    setting = settings[i_figure]
    n_dimension = setting['n_dimension']
    max_degree = setting['max_degree']
    coefficient_value = None
    estimator_type = 'proposed'
    lag_set = [np.ones(n_dimension, dtype=int)]

    if i_figure == '5':
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(0, 11, 1)
        averager_type = 'linear'
        coefficient_type = 'zero'

    elif i_figure == '6':
        other_settings = [
            np.array([0, 1/2-1/2]),
            np.array([0, 1/2-1/2**2]),
            np.array([0, 1/2-1/2**3]),
            np.array([0, 1/2-1/2**4]),
        ]
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(0, 31, 1)
        averager_type = 'linear'
        coefficient_type = 'custom'
        coefficient_value = other_settings[i_curve]
    
    elif i_figure in ['9a', '9b', '9c']:
        other_settings = [
            'linear',
            'do & lee & lozano',
            'kay',
            'lovell & williamson'
        ]
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(0, 31, 1)
        averager_type = other_settings[i_curve]
        coefficient_type = 'random'

    elif i_figure in ['10a', '10b', '10c', '10d', '10e', '10f', '10g', '10h', '10i', '10j', '10k', '10l']:
        signal_size = np.array(setting['signal_size_set'][i_curve])
        snrdb = np.arange(0, 31, 1)
        averager_type = 'do & lee & lozano'
        coefficient_type = 'random'
    
    elif i_figure == '11':
        signal_size = np.array(setting['signal_size_set'][i_curve])
        snrdb = np.arange(0, 31, 1)
        averager_type = 'do & lee & lozano'
        coefficient_type = 'random'
    
    elif i_figure in ['12a', '12b', '12c']:
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(-10, 31, 1)
        averager_type = 'do & lee & lozano'
        lag_set = [2**i_curve * np.ones(n_dimension, dtype=int)]
        coefficient_type = 'zero'
    
    elif i_figure in ['13a', '13b', '13c']:
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(-10, 31, 1)
        averager_type = 'do & lee & lozano'
        if i_curve == 0:
            lag_set = [np.ones(n_dimension, dtype=int)]
        else:
            lag_set = [np.ones(n_dimension, dtype=int),
                       2 * np.ones(n_dimension, dtype=int),
                       2**2 * np.ones(n_dimension, dtype=int),
                       2**3 * np.ones(n_dimension, dtype=int),
                       2**4 * np.ones(n_dimension, dtype=int)]
        coefficient_type = 'random'

    elif i_figure in ['14a', '14b', '14c']:
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(-20, 21, 1)
        averager_type = 'do & lee & lozano'
        if i_curve == 0:
            lag_set = [np.ones(n_dimension, dtype=int)]
        else:
            lag_set = [np.ones(n_dimension, dtype=int),
                       2 * np.ones(n_dimension, dtype=int),
                       2**2 * np.ones(n_dimension, dtype=int),
                       2**3 * np.ones(n_dimension, dtype=int),
                       2**4 * np.ones(n_dimension, dtype=int)]
        coefficient_type = 'random'

    elif i_figure in ['15a', '15b', '15c', '15d', '15e', '15f']:
        signal_size = np.array(setting['signal_size_set'][0])
        snrdb = np.arange(0, 31, 1)
        averager_type = 'do & lee & lozano'
        if i_curve == 1:
            estimator_type = 'DPT'
        elif i_curve == 2:
            estimator_type = 'DPT'
            lag_set = [round(signal_size[0]/max_degree) * np.ones(n_dimension, dtype=int)]
        coefficient_type = 'zero'

    basis_type = 'binomial'        
    if i_figure == '5':
        n_trials = 10
        stopping_condition = 'wrapping'
    else:
        n_trials = 100
        stopping_condition = 'sample'

    if i_figure == '11':
        degree_set = [np.array([max_degree])]
    else:
        degree_set = generate_degree_set(n_dimension, max_degree)

    basis_set = generate_basis_set(signal_size, degree_set, basis_type)
    signal_basis = SignalBasis(signal_size, degree_set, basis_set)
    algorithm_parameter = {
        'signal_basis': signal_basis,
        'lag_set': lag_set,
        'averager_type': averager_type,
        'is_naive': False,
        'estimator_type': estimator_type
    }
    simulation_parameter = {
        'snrdb': snrdb,
        'n_trials': n_trials,
        'coefficient_type': coefficient_type,
        'coefficient_value': coefficient_value,
        'stopping_condition': stopping_condition
    }
    return algorithm_parameter, simulation_parameter