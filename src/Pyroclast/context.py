"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: state.py
Description: Class to store the state of the simulation, including time, iteration, and any additional
constant parameters that are required by the model.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import weakref

class Context:
    """
    Context class to store the context of the simulation, including the grid, pool, model, and state.

    This is passed to each class in the simulation to provide access to the grid, pool, model, and state objects.
    
    It is important to use weak references to avoid circular references and prevent memory leaks.
    Specifically, we should expect a user to write e.g. self.grid = self.ctx.grid in order to avoid
    using .ctx.grid directly. This way, the self.grid will be a weak reference to the grid object, and
    the grid object will be properly garbage collected when it is no longer needed.
    """
    def __init__(self, grid, pool, model, params, options):
        self.grid = weakref.proxy(grid)
        self.pool = weakref.proxy(pool)
        self.model = weakref.proxy(model)
        self.params = weakref.proxy(params)
        self.options = weakref.proxy(options)

class Parameters(dict):
    """
    Class to store the parameters of the simulation, including time, iteration, and any additional
    parameters.
    """
    def __init__(self, args={}):
        # Initialize state parameters
        super().__init__(args)
        
    def _err(self, name):
        return (f"Required parameter '{name}' not present in 'Parameters' object.\n"
                "Make sure it is defined in the input file "
                "or correctly set up in the model.")
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            raise AttributeError(self._err(key))
    
    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(self._err(key))