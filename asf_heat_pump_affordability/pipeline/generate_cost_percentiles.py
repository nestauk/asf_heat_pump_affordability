import pandas as pd
import numpy as np


def generate_dict_cost_percentiles_by_archetype(
    cost_series: pd.Series,
    archetypes_masks: dict,
    percentiles: list or np.array,
) -> dict:
    """
    Generate dict with archetype name and values for cost at each given percentile for every archetype in `archetype_masks`.

    Args
        cost_series (pd.Series): series of (adjusted) cost values
        archetype_masks (dict): dictionary of archetype labels with corresponding pandas Series of boolean values
        percentiles (list or np.array): percentile(s) at which to extract cost values (range 0 to 1)

    Returns
        dict: where archetype names are keys and array of cost values at given percentile(s) are values
    """
    archetype_costs_dict = {}
    for archetype_name, archetype_mask in archetypes_masks.items():
        archetype_costs_dict[archetype_name] = (
            cost_series[archetype_mask].quantile(percentiles).values
        )

    return archetype_costs_dict


def generate_df_cost_percentiles_by_archetype_formatted(
    archetype_costs_dict: dict, percentiles: list or np.array, ref_year: int
) -> pd.DataFrame:
    """
    Convert dict of archetype costs to dataframe.

    Args
        archetype_costs_dict (dict): where archetype names are keys and array of cost values at given percentile(s) are values
        percentiles (list or np.array): percentile(s) at which cost values have been extracted (range 0 to 1)
        ref_year (int): reference year for adjusted costs

    Returns:
        pd.DataFrame: cost percentiles by property archetype

    """
    _cols = _generate_list_column_names(percentiles=percentiles, ref_year=ref_year)
    _cols.insert(0, "property_archetype")

    df = pd.DataFrame(archetype_costs_dict).T.reset_index()
    df.columns = _cols
    df = df.set_index("property_archetype").applymap(_round_cost)

    return df


def _round_cost(v):
    """
    Args v: int or float
    Returns int: value rounded to nearest 10
    """
    return int(round(v, -1))


def _generate_list_column_names(percentiles, ref_year):
    """
    Create list of string column names from percentiles

    Args
        percentiles (list or np.array): percentile(s) at which cost values have been extracted (range 0 to 1)
        cpi_data_year (int): reference year for adjusted costs

    Returns
        list: column names
    """
    columns = []
    mapping = {
        0: "min",
        0.25: "lower_quartile",
        0.5: "median",
        0.75: "upper_quartile",
        1: "max",
    }

    for pc in percentiles:
        if pc in [0, 0.25, 0.5, 0.75, 1]:
            columns.append(mapping[pc])
        else:
            columns.append(f"{int(pc * 100)}th_percentile")

    columns = ["_".join([col, "adjusted_cost", str(ref_year)]) for col in columns]

    return columns
