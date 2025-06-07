"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: banner.py
Description: Banner to be printed at the start of the simulation.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
    

if __name__ == '__main__':
    print_banner()