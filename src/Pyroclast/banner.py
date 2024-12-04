"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: banner.py
Description: Banner to be printed at the start of the simulation.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari. All rights reserved.

See LICENSE file in the project root for full license information.
"""
import termcolor

banner = termcolor.colored(r"""
 ______   ______   ___   ____ _        _    ____ _____ 
|  _ \ \ / /  _ \ / _ \ / ___| |      / \  / ___|_   _|
| |_) \ V /| |_) | | | | |   | |     / _ \ \___ \ | |  
|  __/ | | |  _ <| |_| | |___| |___ / ___ \ ___) || |  
|_|    |_| |_| \_\\___/ \____|_____/_/   \_\____/ |_|  (v0.1)
""", 'red', attrs=['bold'])
                               
def print_banner():
    print(banner)
    

