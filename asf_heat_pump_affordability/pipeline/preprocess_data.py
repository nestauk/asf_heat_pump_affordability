import numpy as np
import pandas as pd


def apply_exclusion_criteria(
    df: pd.DataFrame, cost_year_min: int = None, cost_year_max: int = None
) -> pd.DataFrame:
    """
    Apply exclusion criteria to joined MCS-EPC dataframe to get analytical sample.

    Exclusion criteria are rows where:
        - Cost is NaN
        - Tech type is NaN
        - Tech type != Air Source Heat Pump
        - No joined EPC record
        - Insufficient property data to identify property archetype

    Args
        df (pd.DataFrame): joined MCS-EPC dataset
        cost_year_min (int): min year of heat pump installation cost data to include
        cost_year_max (int): max year of heat pump installation cost data to include

    Returns
        pd.DataFrame: Joined MCS and EPC dataset with analytical exclusion criteria applied.
    """
    if cost_year_min:
        df = df[df["commission_year"] >= cost_year_min]
    if cost_year_max:
        df = df[df["commission_year"] <= cost_year_max]

    df = df.replace("(?i)unknown", np.nan, regex=True)

    key_variables = [
        "cost",
        "tech_type",
        "original_epc_index",
        "CONSTRUCTION_AGE_BAND",
        "BUILT_FORM",
        "PROPERTY_TYPE",
    ]
    df = df.dropna(subset=key_variables, how="any")
    df = df[df["tech_type"] == "Air Source Heat Pump"]

    return df
