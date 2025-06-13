"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: timer.py
Description: Simple timer class to profile the execution time of the code.

Author: Marcel Ferrari
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import time
from collections import OrderedDict
from tabulate import tabulate
from logging import get_logger

logger = get_logger(__name__)

class Timer:
    def __init__(self):
        self.data = OrderedDict()

    def start(self, category, operation):
        if category not in self.data:
            self.data[category] = {'total': 0, 'operations': {}}
        if operation not in self.data[category]['operations']:
            self.data[category]['operations'][operation] = {'start': None, 'elapsed': 0}
        
        self.data[category]['operations'][operation]['start'] = time.perf_counter()

    def stop(self, category, operation):
        if category in self.data and operation in self.data[category]['operations']:
            start_time = self.data[category]['operations'][operation]['start']
            if start_time is not None:
                elapsed = time.perf_counter() - start_time
                self.data[category]['operations'][operation]['elapsed'] += elapsed
                self.data[category]['total'] += elapsed
                self.data[category]['operations'][operation]['start'] = None

    def report(self):
        """Generate a structured report detailing timings for each category and operation."""
        logger.info("\nPerformance Report:")
        for category, cat_data in self.data.items():
            total_cat_time = cat_data['total']
            logger.info(f"\n{category}")

            # Prepare table data for operations sorted by elapsed time (descending)
            table_data = []
            sorted_operations = sorted(
                cat_data['operations'].items(),
                key=lambda x: x[1]['elapsed'],
                reverse=True
            )

            for operation, op_data in sorted_operations:
                op_time = op_data['elapsed']
                percentage = (op_time / total_cat_time) * 100 if total_cat_time > 0 else 0
                table_data.append([operation, f"{op_time:.4f} sec", f"{percentage:.2f}%"])
            else: # Append total time to the end of the table
                table_data.append(["Total", f"{total_cat_time:.4f} sec", "100.00%"])

            # Print table using tabulate
            logger.info(tabulate(table_data, headers=["Operation", "Time", "Percentage"], tablefmt="simple"))

    def time_function(self, category, operation):
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.time_section(category, operation):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def time_section(self, category, operation):
        class ProfileContextManager:
            def __init__(inner_self):
                inner_self.category = category
                inner_self.operation = operation

            def __enter__(inner_self):
                self.start(inner_self.category, inner_self.operation)
                return inner_self

            def __exit__(inner_self, exc_type, exc_value, traceback):
                self.stop(inner_self.category, inner_self.operation)

        return ProfileContextManager()
