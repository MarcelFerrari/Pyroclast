"""
Pyroclast: Scalable Geophysics Models
https://github.com/MarcelFerrari/Pyroclast

File: benchmark/utils.py
Description: Contians utils needed to deal with date and time of benchmrks

Author: Alexander Sotoudeh
Copyright (c) 2024 Marcel Ferrari.

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""     

import datetime

from pydantic import AfterValidator
from annotated_types import Annotated


def datetime_factory() -> datetime.datetime:
    """
    Construct a tz aware, datetime object with timezone UTC
    """
    return datetime.datetime.now(datetime.timezone.utc)


# Shorthand
dtf = datetime_factory


def localize_time(dt: datetime.datetime) -> datetime.datetime:
    """
    Convert any datetime object to UTC (Account for locales not set the same across nodes)
    """
    if dt.tzinfo is None:
        raise TypeError("Datetime needs to be timezone-aware")

    if dt.tzinfo.utcoffset(dt) != datetime.timezone.utc:
        return dt.astimezone(datetime.timezone.utc)

    return dt


# Define custom datetime that is always UTC
DateTimeUTC = Annotated[datetime.datetime, AfterValidator(localize_time)]
