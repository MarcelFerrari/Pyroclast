"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: format.py
Description: Utility functions for pretty formatting of numbers and strings.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

def s2yr(seconds):
    """
    Format a time duration in seconds into yr, Myr, or Gyr.
    
    Parameters:
        seconds (float): The time duration in seconds.
    
    Returns:
        str: A formatted string with the time in the most appropriate unit.
    """
    # Constants
    SECONDS_IN_YEAR = 365.25 * 24 * 60 * 60  # Seconds in one year
    
    # Convert to years
    years = seconds / SECONDS_IN_YEAR

    # Determine the unit and value
    if years >= 1e9:  # Billions of years
        value = years / 1e9
        unit = "Gyr"
    elif years >= 1e6:  # Millions of years
        value = years / 1e6
        unit = "Myr"
    else:  # Less than a million years
        value = years
        unit = "yr"

    # Format the result with two decimal places
    return f"{value:.2f} {unit}"