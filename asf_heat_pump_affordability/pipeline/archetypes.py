import pandas as pd
from asf_heat_pump_affordability import config


def classify_dict_archetypes_masks(
    df: pd.DataFrame,
    ages_mapping: dict = config["construction_age_band_1950_split"],
    year_label: str = "1950",
) -> dict:
    """
    Generate dictionary where keys are property archetype labels and values are corresponding pandas Series of boolean
    values which can be used to mask or filter the dataframe for the given archetype.

    Produces eight property archetypes: flats; semi-detached & terraced houses and maisonettes; detached houses; and
    bungalows; with each group split into 'pre' and 'post' groupings for a given construction year determined by
    `ages_mapping`.

    Args
        df (pd.DataFrame): joined MCS-EPC dataframe containing property characteristics
        ages_mapping (dict): dict mapping construction age bands to boolean values. Default dict where construction ages
                             before 1950 are mapped to `True` and after 1950 to `False`.
        year_label (str): bisecting year for 'pre' and 'post' construction age band groupings to label data with.
                          Defaults to "1950".

    Returns
        dict: dictionary of archetype labels with corresponding pandas Series of boolean values

    """
    age_split = df["CONSTRUCTION_AGE_BAND"].map(ages_mapping)

    masks = {}

    # Flats
    masks[f"pre_{year_label}_flat"] = (df["PROPERTY_TYPE"] == "Flat") & age_split
    masks[f"post_{year_label}_flat"] = (df["PROPERTY_TYPE"] == "Flat") & (~age_split)

    # Semi-detached & terraced houses, and maisonettes
    masks[f"pre_{year_label}_semi_terraced_house"] = (
        _classify_series_semi_terraced_house(df) & age_split
    )
    masks[
        f"post_{year_label}_semi_terraced_house"
    ] = _classify_series_semi_terraced_house(df) & (~age_split)

    # Detached houses
    masks[f"pre_{year_label}_detached_house"] = (
        _classify_series_detached_house(df) & age_split
    )
    masks[f"post_{year_label}_detached_house"] = _classify_series_detached_house(df) & (
        ~age_split
    )

    # Bungalows
    masks[f"pre_{year_label}_bungalow"] = (
        df["PROPERTY_TYPE"] == "Bungalow"
    ) & age_split
    masks[f"post_{year_label}_bungalow"] = (df["PROPERTY_TYPE"] == "Bungalow") & (
        ~age_split
    )

    return masks


def _classify_series_semi_terraced_house(df: pd.DataFrame) -> pd.Series:
    """
    Generate pandas Series of boolean values to classify whether or not properties are in the category of 'semi-detached
    or terraced houses, or maisonettes'.

    Args
        df (pd.DataFrame): joined MCS-EPC dataframe containing property characteristics

    Returns
        pd.Series: boolean values where `True` indicates the property is a semi-detached or terraced house, or maisonette
    """
    mask = (df["PROPERTY_TYPE"] == "Maisonette") | (
        (df["PROPERTY_TYPE"] == "House")
        & (
            df["BUILT_FORM"].isin(
                [
                    "Semi-Detached",
                    "Mid-Terrace",
                    "End-Terrace",
                    "Enclosed End-Terrace",
                    "Enclosed Mid-Terrace",
                ]
            )
        )
    )
    return mask


def _classify_series_detached_house(df: pd.DataFrame) -> pd.Series:
    """
    Generate pandas Series of boolean values to classify whether or not properties are detached houses.

    Args
        df (pd.DataFrame): joined MCS-EPC dataframe containing property characteristics

    Returns
        pd.Series: boolean values where `True` indicates the property is a detached house
    """
    return (df["PROPERTY_TYPE"] == "House") & (df["BUILT_FORM"] == "Detached")
