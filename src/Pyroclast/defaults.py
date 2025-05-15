def default_config():
    # Create a default configuration
    default_state = {
        'time' : 0.0,
        'iteration' : 0
    }

    default_params = {
        'cfl_dispmax' : 0.5,
    }

    default_options = {
        'threading_layer' : 'omp',
        'num_threads' : 0,
        'checkpoint_interval' : 25,
        'checkpoint_file' : 'checkpoint.pkl',
        'framedump_interval' : 1,
    }

    return default_state, default_params, default_options
