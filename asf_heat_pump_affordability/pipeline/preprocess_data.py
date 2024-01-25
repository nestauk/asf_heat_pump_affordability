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


def generate_df_adjusted_costs(mcs_epc_df, cpi_quarters_df):
    """
    Join CPI (consumer price index) dataframe containing quarterly adjustment factors to MCS-EPC dataframe and
    calculate adjusted installation costs for each row.

    Args
        mcs_epc_df (pd.DataFrame): joined MCS-EPC dataframe
        cpi_quarters_df (pd.DataFrame): quarterly CPI data with adjustment factors for each quarter

    Returns
        pd.DataFrame: MCS-EPC dataframe with CPI values, adjustment factors, and adjusted costs
    """
    mcs_epc_df["year_quarter"] = _generate_series_year_quarters(
        commission_date_series=mcs_epc_df["commission_date"]
    )

    mcs_epc_inf = mcs_epc_df.merge(
        cpi_quarters_df, how="left", left_on="year_quarter", right_on="Title"
    )

    mcs_epc_inf["adjusted_cost"] = (
        mcs_epc_inf["cost"] * mcs_epc_inf["adjustment_factor"]
    )

    return mcs_epc_inf


def _generate_series_year_quarters(commission_date_series):
    """
    Generate a series of years and quarters from a series of dates.

    Args
        commission_date_series (pd.Series): commission dates with year, month, and day

    Returns
        pd.Series: series of year and quarter values in the form `YYYY QN`
    """
    return (
        commission_date_series.pipe(pd.to_datetime).dt.year.astype(str)
        + " Q"
        + commission_date_series.pipe(pd.to_datetime).dt.quarter.astype(str)
    )
