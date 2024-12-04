"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solver.py
Description: Main solver class for Pyroclast, which handles the time integration of the model.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""
import toml
import importlib
import os
import re
import pickle
import time
import numpy as np
import numba as nb

from Pyroclast.context import Context, Parameters
from Pyroclast.profiling import timer
from Pyroclast.banner import print_banner
import Pyroclast.format as fmt

class Pyroclast():
    def __init__(self, args):
        # Print Pyroclast banner
        print_banner()

        # Initialize constants
        self.input_file = args['input']
        
        # Load input file
        # This creates the self.options and self.params dictionaries
        self.load_input(self.input_file) 

        # Update options with command line arguments
        # This allows us to override options from the input file
        self.options.update(args)

        # Set up threading layer
        threading_layer = self.options.get('threading_layer', 'omp')
        nb.config.THREADING_LAYER = threading_layer
        print(f"Using threading layer: {threading_layer}")

        # Set up number of threads
        num_threads = self.options.get('num_threads', 0)
        if num_threads > 0:
            nb.set_num_threads(num_threads)
        num_threads = nb.get_num_threads()
        print(f"Numba using {num_threads} threads.")
        
        # Initialize solver parameters
        self.grid = self.load_obj('grid')
        self.pool = self.load_obj('pool')
        self.model = self.load_obj('model')

        # Set context
        ctx = Context(self.grid,
                      self.pool,
                      self.model,
                      self.params,
                      self.options)

        # Pass context to grid, pool, and model
        # We pass proxy objects to avoid circular references
        # and prevent memory leaks
        self.grid.set_context(ctx)
        self.pool.set_context(ctx)
        self.model.set_context(ctx)

        # Read checkpoint options
        self.checkpoint_interval = self.options.get('checkpoint_interval', 25)
        self.checkpoint_file = self.options.get('checkpoint_file', 'checkpoint.pkl')

    def load_input(self, input):
        # Check if input file exists
        assert os.path.exists(input), "Input file not found."

        with open(input, 'r') as f:
            config = toml.load(f)

        self.options = Parameters(config.get('options', {}))
        self.params = Parameters(config.get('params', {}))

        # Check if options and params are not empty
        assert self.options, "Please provide an [options] annotation in the input file."
        assert self.params, "Please provide a [params] annotation in the input file."
    
    def load_obj(self, name):
        assert name in {'grid', 'pool', 'model'}, f"Invalid object name: {name}"
        assert name in self.options, f"Please provide a {name} option in the input file."
        
        # Parse package and class name
        # The format is package.class
        # Check that the format is correct
        def validate_format(attr):
            # Regular expression pattern to match "package.classname" format
            pattern = r"^[a-zA-Z_]\w*\.[a-zA-Z_]\w*$"
            return bool(re.match(pattern, attr))
        
        assert validate_format(self.options[name]), \
               f"Invalid format for {name} option. Please use 'package.class' format."
        
        package, class_name = self.options[name].split('.')
        module = importlib.import_module(f"Pyroclast.{name}.{package}")
        model_class = getattr(module, class_name)

        return model_class()

    def write_checkpoint(self, file_path):
        """
        Pickle grid, pool, model and params objects to a file.
        """
        state = {
            'grid': self.grid,
            'pool': self.pool,
            'model': self.model,
            'options': self.options,
            'params': self.params
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)
    
    def load_checkpoint(self, file_path):
        """
        Load grid, pool, model, and state objects from a file.
        """        
        with open(file_path, 'rb') as f:
            state = pickle.load(f)

        self.grid = state['grid']
        self.pool = state['pool']
        self.model = state['model']
        self.params = state['params']
        self.options = state['options']
        
        # Update the context of each object
        self.ctx = Context(self.grid,
                           self.pool,
                           self.model,
                           self.params,
                           self.options)
        
        self.grid.set_context(self.ctx)
        self.pool.set_context(self.ctx)
        self.model.set_context(self.ctx)
        
    def solve(self):
        print("Starting simulation...")

        # Reload variables from checkpoint
        if os.path.exists(self.checkpoint_file):
            print("Attempting to load checkpoint...")
            print(f"Loading checkpoint from {self.checkpoint_file}.")
            self.load_checkpoint(self.checkpoint_file)
            print(f"Resuming simulation at t = {fmt.s2yr(self.params.t)}")
            
            # Reset iteration counter
            self.params.it = 0
        else:
            print("Checkpoint file not found. Initializing simulation from scratch.")
            # Initialize state variables
            self.params.dt = self.params.get('dt', 0.0)
            self.params.t = 0.0
            self.params.it = 0

            self.params.t_end = self.params.get('t_end', np.inf)
            self.params.max_it = self.params.get('max_iter', np.inf)

            assert self.params.t_end != np.inf or self.params.max_it != np.inf, \
                     "Please provide either a t_end or max_it parameter in the input file."

            # Initialize grid, pool, and model
            self.grid.initialize()
            self.pool.initialize()
            self.model.initialize()

        print(f"Saving model every {self.checkpoint_interval} timesteps.")

        # Main time integration loop
        print("Starting time integration loop...")
        start = time.time()
        while self.params.t <= self.params.t_end and self.params.it < self.params.max_it:
            # 1) Interpolate marker values to grid nodes
            with timer.time_section("Main Loop", "Interpolation"):
                self.model.interpolate()

            # 2) Solve the model equations
            with timer.time_section("Main Loop", "Model Solve"):
                self.model.solve()

            # 3) Update the pool of markers with new values
            with timer.time_section("Main Loop", "Interpolation"):
                self.pool.interpolate()

            # 4) Update time step
            self.model.update_time_step()

            # 5) Advect markers
            with timer.time_section("Main Loop", "Advection"):
                self.pool.advect()

            # 6) Update time and iteration counter
            self.params.t += self.params.dt
            self.params.it += 1

            # 7) Write checkpoint
            if (self.params.it+1) % self.checkpoint_interval == 0:
                with timer.time_section("Main Loop", "Checkpoint I/O"):
                    print(f"Writing checkpoint to {self.checkpoint_file}.")
                    self.write_checkpoint(self.checkpoint_file)

            # 8) Print progress every 5% of the simulation
            if (self.params.it + 1) % (max(self.params.max_it // 20, 1)) == 0:
                print(f"Progress: {100*self.params.it/self.params.max_it:.2f}%, it = {self.params.it}, t = {fmt.s2yr(self.params.t)}, dt = {fmt.s2yr(self.params.dt)}")
        
        print("Time loop complete!")
        # End of time integration loop
        end = time.time() 
        
        # Finalize grid, pool, and model
        print("Finalizing simulation...")
        self.grid.finalize()
        self.pool.finalize()
        self.model.finalize()
        
        timer.report()
        print(f"\nTotal workload runtime {end-start:.4f} seconds.")



