"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: solver.py
Description: Main solver class for Pyroclast, which handles the time integration of the model.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
import toml
import importlib
import os
import re
import pickle
import time
import numpy as np
import numba as nb

from Pyroclast.context import Context, ContextNamespace
from Pyroclast.profiling import timer
from Pyroclast.banner import print_banner
import Pyroclast.format as fmt
from Pyroclast.defaults import default_config

class Pyroclast():
    def __init__(self, args):
        # Print Pyroclast banner
        print_banner()

        # Initialize constants
        self.input_file = args['input']
        
        # Load input file
        state, params, options = self.load_input(self.input_file) 

        # Read checkpoint options
        chkpt_file = options.checkpoint_file
        if os.path.exists(chkpt_file):
            print(f"Checkpoint file {chkpt_file} found. Loading checkpoint...")
            # Checkpoint file found -> load context from checkpoint
            with open(chkpt_file, 'rb') as f:
                chkpt = pickle.load(f)

            self.ctx = chkpt['ctx']
            self.grid = chkpt['grid']
            self.pool = chkpt['pool']
            self.model = chkpt['model']
            print("Checkpoint loaded successfully.")
        else:
            print(f"No checkpoint file found. Starting fresh simulation.")
            # Create context object
            self.ctx = Context(state, params, options)

            # Initialize fresh solver components
            self.grid = self.load_component('grid', options['grid'])
            self.pool = self.load_component('pool', options['pool'])
            self.model = self.load_component('model', options['model'])

        # Update options with command line arguments
        # This allows us to override some options from the input file
        self.ctx.options.update(args)
        
    def load_input(self, input):
        # Load TOML input file
        with open(input, 'r') as f:
            config = toml.load(f)

        # Load defaults
        state, params, options = default_config()

        # Update params and options with custom values
        params.update(config.get('parameters', {}))
        options.update(config.get('options', {}))

        return ContextNamespace(state),  \
               ContextNamespace(params), \
               ContextNamespace(options) \
    
    def set_threading(self, options):
        # Set up threading layer
        tl = options.threading_layer
        print(f"Using threading layer: {tl}")
        nb.config.THREADING_LAYER = tl
        
        # Set up number of threads
        num_threads = options.num_threads
        if num_threads > 0:
            nb.set_num_threads(num_threads)
        num_threads = nb.get_num_threads()
        print(f"Numba using {num_threads} threads.")
        
    def load_component(self, comp, cls_name):
        # Check that the format is correct
        def validate_format(attr):
            # Regular expression pattern to match "package.classname" format
            pattern = r"^[a-zA-Z_]\w*\.[a-zA-Z_]\w*$"
            return bool(re.match(pattern, attr))
        
        assert validate_format(cls_name), \
               f"Invalid format for {cls_name} option. Please use 'module.class' format."
        
        module, cls = cls_name.split('.')
        module = importlib.import_module(f"Pyroclast.{comp}.{module}")
        cls = getattr(module, cls)
        return cls(self.ctx)

    def write_checkpoint(self, file_path):
        """
        Pickle context, grid, pool, and model objects to a file.
        """
        print(f"Writing checkpoint to {file_path}...")
        chkpt = {
            'ctx': self.ctx,
            'grid': self.grid,
            'pool': self.pool,
            'model': self.model,
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(chkpt, f)

        print(f"Checkpoint written to {file_path}.")

    def solve(self):
        # Read options
        s, p, o = self.ctx
        
        print("Starting simulation...")
        print(f"Saving model every {o.checkpoint_interval} timesteps.")

        # Load initial timestep dt into state
        s.dt = p.dt_initial

        # Read max_iterations
        assert 'max_iterations' in p, \
                "Missing 'max_iterations' parameter in input file."
        assert p.max_iterations > 0, \
                "max_iterations must be greater than 0."
    
        
        # Main time integration loop
        print("Starting time integration loop...")
        
        # Compute zfill padding
        zpad = len(str(p.max_iterations//o.framedump_interval)) + 1
        frame = 0

        start = time.time()
        while s.iteration < p.max_iterations:
            # 1) Interpolate marker values to grid nodes
            with timer.time_section("Main Loop", "Interpolation"):
                self.grid.interpolate(self.ctx)

            # 2) Solve the model equations
            with timer.time_section("Main Loop", "Model Solve"):
                self.model.solve(self.ctx)

            # 3) Update the pool of markers with new values
            with timer.time_section("Main Loop", "Interpolation"):
                self.pool.interpolate(self.ctx)

            # 4) Update time step
            self.model.update_time_step(self.ctx)

            # 5) Advect markers
            with timer.time_section("Main Loop", "Advection"):
                self.pool.advect(self.ctx)

            # 6) Update time
            s.time += s.dt
            
            # 7) Write Data
            if (s.iteration % o.framedump_interval) == 0:
                # Dump state to file
                with open(f"frame_{str(frame).zfill(zpad)}.pkl", 'wb') as f:
                    pickle.dump(self.ctx.to_dict(), f)
                print(f"Frame {frame} written to file.")
                frame += 1 # Increment frame counter

            if ((s.iteration+1) % o.checkpoint_interval) == 0:
                self.write_checkpoint(o.checkpoint_file)
            
            if ((s.iteration+1) % max(p.max_iterations // 20, 1)) == 0:
                percent = 100 * s.iteration / p.max_iterations
                it = s.iteration
                t = fmt.s2yr(s.time)
                dt = fmt.s2yr(s.dt)
                print(f"Progress: {percent:.2f}%, it = {it}, t = {t}, dt = {dt}")

            # 8) Increment iteration
            s.iteration += 1
        
        # End of time integration loop
        end = time.time() 

        print("Time loop complete!")
        
        # Finalize grid, pool, and model
        print("Finalizing simulation...")
        self.grid.finalize(self.ctx)
        self.pool.finalize(self.ctx)
        self.model.finalize(self.ctx)
        
        timer.report()
        print(f"\nTotal workload runtime {end-start:.4f} seconds.")


