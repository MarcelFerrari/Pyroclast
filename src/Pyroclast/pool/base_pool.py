"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: base_pool.py
Description: Base pool class for Pyroclast, which handles the pool of markers.
             This class is meant to be inherited by specific pool implementations and
             serves to implement the required methods for the pool class.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

class BasePool:
    def __init__(self, ctx):
        raise NotImplementedError()
    
    def interpolate(self, ctx):
        raise NotImplementedError()
    
    def advect(self, ctx):
        raise NotImplementedError()
    
    def finalize(self, ctx):
        pass
