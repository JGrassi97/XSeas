import os
import xarray as xr
import datetime

def load_variables(base_path, variables, variables_codes):

    dataset = []
    for variable, code in zip(variables, variables_codes):
            path = os.path.join(base_path, variable, 'final.nc')

            try:
                dat = xr.open_dataset(path)[code].mean('plev')
            except:
                dat =xr.open_dataset(path)[code]

            dataset.append(dat)

    return dataset



def get_ordinal_suffix(day):
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
        

def day_of_year_to_date(day_of_year, year=None):
    if year is None:
        year = datetime.datetime.now().year
    
    try:
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_of_year - 1)
        day = date.day
        month = date.strftime("%b")
        ordinal_suffix = get_ordinal_suffix(day)
        return fr"{month} {day}{ordinal_suffix}"
    except ValueError:
        return "Invalid day of the year"