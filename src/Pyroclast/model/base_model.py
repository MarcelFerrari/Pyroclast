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
    def __init__(self, ctx):
        raise NotImplementedError()
        
    def finalize(self, ctx):
        pass

    def solve(self, ctx):
        raise NotImplementedError()
    
    def update_time_step(self, ctx):
        raise NotImplementedError()