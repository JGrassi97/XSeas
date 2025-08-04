"""
General utility functions for XSeas package.

This module provides utility functions for date formatting and other
common operations used throughout the package.
"""

import datetime
from typing import Optional

def get_ordinal_suffix(day: int) -> str:
    """
    Get the ordinal suffix for a given day of the month.
    
    Parameters
    ----------
    day : int
        Day of the month (1-31).
        
    Returns
    -------
    str
        Ordinal suffix ('st', 'nd', 'rd', or 'th').
        
    Examples
    --------
    >>> get_ordinal_suffix(1)
    'st'
    >>> get_ordinal_suffix(22)
    'nd'
    >>> get_ordinal_suffix(13)
    'th'
    """
    if 10 <= day <= 20:
        return 'th'
    
    last_digit = day % 10
    suffix_map = {1: 'st', 2: 'nd', 3: 'rd'}
    return suffix_map.get(last_digit, 'th')


def day_of_year_to_date(day_of_year: int, year: Optional[int] = None) -> str:
    """
    Convert a day of the year to a formatted date string.
    
    Parameters
    ----------
    day_of_year : int
        Day of the year (1-365 or 1-366 for leap years).
    year : Optional[int], default=None
        Year to use for conversion. If None, uses current year.
        
    Returns
    -------
    str
        Formatted date string (e.g., "Jan 1st", "Dec 25th").
        Returns "Invalid day of the year" if conversion fails.
        
    Examples
    --------
    >>> day_of_year_to_date(1)
    'Jan 1st'
    >>> day_of_year_to_date(359, 2023)
    'Dec 25th'
    """
    if year is None:
        year = datetime.datetime.now().year
    
    try:
        # Convert day of year to actual date
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
        
        # Format components
        day = date.day
        month = date.strftime("%b")
        suffix = get_ordinal_suffix(day)
        
        return f"{month} {day}{suffix}"
        
    except (ValueError, OverflowError):
        return "Invalid day of the year"


def validate_day_of_year(day_of_year: int, year: Optional[int] = None) -> bool:
    """
    Validate if a day of year is valid for the given year.
    
    Parameters
    ----------
    day_of_year : int
        Day of the year to validate.
    year : Optional[int], default=None
        Year to check against. If None, uses current year.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    if year is None:
        year = datetime.datetime.now().year
    
    try:
        datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
        return True
    except (ValueError, OverflowError):
        return False


# Backward compatibility aliases
_get_ordinal_suffix = get_ordinal_suffix  # Keep private version for compatibility