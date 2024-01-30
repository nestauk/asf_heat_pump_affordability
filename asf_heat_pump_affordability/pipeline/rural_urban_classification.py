import pandas as pd
from asf_heat_pump_affordability.getters import get_data


def generate_df_rural_class_gb() -> pd.DataFrame:
    """
    Get dataframe of 2-fold rural/urban classification by postcode for Great Britain.

    Returns
        pd.DataFrame: rural/urban classification by postcode for Great Britain
    """
    rural_class_engwal = join_df_postcode_to_rural_class_engwal()
    rural_class_sct = get_data.get_df_rural_class_sct()

    df = rural_class_engwal.merge(
        rural_class_sct, how="outer", on=["postcode", "OA11CD"]
    )
    df["rural_urban_class_2fold"] = df[
        "OA11_rural_urban_classification_2011_(2_fold)"
    ].fillna(df["sct_rural_urban_class_2fold"])

    return df


def join_df_postcode_to_rural_class_engwal(
    rural_sheet_name: str = "OA11",
    pcd_join: str = "OA11CD",
) -> pd.DataFrame:
    """
    Join rural/urban classification by area with postcode to area dataframe for England and Wales.

    Args
        rural_sheet_name (str): name of sheet in rural-urban classification file where rural/urban classification
                                information for chosen geographic level is stored. Default "OA11" to get rural/urban
                                classification by output area.
        pcd_join (str): name of column in postcode-to-area dataframe to conduct join with rural-urban-classification
                        dataframe on. Default "OA11CD".

    Returns
        pd.DataFrame: rural/urban classification by postcode for England and Wales
    """
    postcode_to_area_df = get_data.get_df_postcode_to_area_engwal()
    rural_class_engwal_df = get_data.get_df_rural_class_engwal(
        sheet_name=rural_sheet_name
    )

    df = postcode_to_area_df.merge(
        rural_class_engwal_df,
        how="left",
        left_on=pcd_join,
        right_on=rural_class_engwal_df.columns[0],
    )

    df = df[["postcode", "OA11CD", "OA11_rural_urban_classification_2011_(2_fold)"]]

    return df
