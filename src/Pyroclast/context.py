"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: state.py
Description: Class to store the state of the simulation, including time, iteration, and any additional
constant parameters that are required by the model.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

class Context:
    """
        Context class to store global simulation state, parameters, and runtime options.

        This object is passed to all major components (e.g. grid, pool, model) to access shared data.
        Components should not store references to each other — all interaction occurs through the context.
    """
    def __init__(self, state = None, params = None, options = None):
        """
        Initialize the context with the given state, parameters, and options.
        state: Dictionary containing the state of the simulation.
        params: Dictionary containing the parameters of the simulation.
        options: Dictionary containing the options of the simulation.
        """
        # Initialize state parameters
        self.state = state or ContextNamespace()
        self.params = params or ContextNamespace()
        self.options = options or ContextNamespace()
    
    def __iter__(self):
        yield self.state
        yield self.params
        yield self.options
    
    def to_dict(self):
        """
        Convert the context to a dictionary.
        """
        return {
            'state': dict(self.state),
            'params': dict(self.params),
            'options': dict(self.options)
        }


class ContextNamespace(dict):
    """
    A dictionary-like class that allows attribute access to its keys.
    """
    def __init__(self, args = None):
        # Initialize state parameters
        super().__init__(args or {})
    
    def _raise(self, key):
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
    
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            self._raise(key)
    
    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            self._raise(key)