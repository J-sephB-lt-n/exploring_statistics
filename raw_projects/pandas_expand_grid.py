
# function equivalent to R's expand_grid():
# (note: requires library 'itertools')

import itertools
import pandas as pd

def expand_grid(data_dict):
    # https://stackoverflow.com/questions/12130883/r-expand-grid-function-in-python
    """Create a dataframe from every combination of given values."""
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())
