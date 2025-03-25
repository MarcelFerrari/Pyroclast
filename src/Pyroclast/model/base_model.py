"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: base_model.py
Description: Base model class for Pyroclast, which handles the model equations.
             This class is meant to be inherited by specific model implementations and
             serves to implement the required methods for the model class.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

class BaseModel:
    def set_context(self, ctx):
        self.ctx = ctx

    def initialize(self):
        raise NotImplementedError(f"Grid class {self.__class__.__name__} must implement initialize method")
        
    def finalize(self):
        raise NotImplementedError(f"Grid class {self.__class__.__name__} must implement finalize method")

    def __getstate__(self):
        # Return a dictionary with only the essential attributes
        state = self.__dict__.copy()
        state.pop('ctx', None)  # Exclude weak references
        return state

    def __setstate__(self, state):
        # Restore the object's state from the unpickled state dictionary
        self.__dict__.update(state)
        self.ctx = None  # Reset weak reference

    def solve(self):
        raise NotImplementedError(f"Model class {self.__class__.__name__} must implement solve method")
    
    def update_time_step(self):
        raise NotImplementedError(f"Model class {self.__class__.__name__} must implement update_time_step method")