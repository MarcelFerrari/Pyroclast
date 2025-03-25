"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: base_grid.py
Description: Base grid class for Pyroclast, which handles the grid data structure.
             This class is meant to be inherited by specific grid implementations and
             serves to implement the required methods for the grid class.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

class BaseGrid:
    def set_context(self, ctx):
        self.ctx = ctx

    def initialize(self):
        raise NotImplementedError(f"Grid class {self.__class__.__name__} must implement initialize method")
        
    def finalize(self):
        pass
    
    def __getstate__(self):
        # Return a dictionary with only the essential attributes
        state = self.__dict__.copy()
        state.pop('ctx', None)  # Exclude weak references
        return state

    def __setstate__(self, state):
        # Restore the object's state from the unpickled state dictionary
        self.__dict__.update(state)
        self.ctx = None  # Reset weak reference

    def interpolate(self):
        raise NotImplementedError(f"Grid class {self.__class__.__name__} must implement interpolate method")