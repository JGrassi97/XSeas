import os
import xarray as xr
import datetime

# -- UTILITY FUNCTIONS FOR DATE FORMATTING -- #
# This function are used to convert a day of the year to a formatted date string.
# Maily intended for use in plotting and labeling.

def _get_ordinal_suffix(day : int) -> str:
    """ Get the ordinal suffix for a given day of the month."""

    if 10 <= day <= 20:
        return 'th'
    else:
        last_digit = day % 10
        if last_digit == 1:
            return 'st'
        elif last_digit == 2:
            return 'nd'
        elif last_digit == 3:
            return 'rd'
        else:
            return 'th'
        

def day_of_year_to_date(day_of_year : int, year : int = None) -> str:
    """ Convert a day of the year to a formatted date string (e.g., "Jan 1st"). """

    if year is None:
        year = datetime.datetime.now().year
    
    try:
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
        day = date.day
        month = date.strftime("%b")
        ordinal_suffix = _get_ordinal_suffix(day)
        return fr"{month} {day}{ordinal_suffix}"
    except ValueError:
        return "Invalid day of the year"