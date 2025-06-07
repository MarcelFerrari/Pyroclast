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