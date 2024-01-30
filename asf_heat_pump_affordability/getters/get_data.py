import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from typing import Optional
import time
from asf_heat_pump_affordability import config


def get_df_from_url(
    url: str, file_type: Optional[str] = None, file_name: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """
    Get dataframe from file stored at URL.

    Args
        url (str): URL location of file download
        **kwargs: pandas.read_csv() kwargs if csv is True else pandas.read_excel() kwargs

    Returns
        pd.DataFrame
    """
    with requests.Session() as session:
        res = session.get(url)
    if file_type == "csv":
        df = pd.read_csv(BytesIO(res.content), **kwargs)
    elif file_type == "zip":
        df = pd.read_csv(ZipFile(BytesIO(res.content)).open(file_name))
    else:
        df = pd.read_excel(BytesIO(res.content), **kwargs)

    return df


def get_list_off_gas_postcodes(sheet_name: str = "Off-Gas Postcodes 2023") -> list:
    """
    Get series of off-gas postcodes.
    """
    df = get_df_from_url(
        url=config["data_source"]["off_gas_postcodes_url"],
        file_type="csv",
        sheet_name=sheet_name,
    )

    off_gas_postcodes = df["Post Code"].str.replace(" ", "").to_list()

    return off_gas_postcodes


def get_dict_rural_class_ew():
    """
    Get dict of dataframes with rural urban classification information for areas at different geographic levels.
    """
    df_dict = get_df_from_url(
        config["data_source"]["rural_urban_classification_2011_url"],
        engine="odf",
        sheet_name=None,
        skiprows=2,
    )

    for sheet, df in df_dict.items():
        df.columns = [
            f"{sheet}_{'_'.join(col.lower().split(' '))}" for col in df.columns
        ]

    return df_dict


def _get_df_postcode_to_area_ew():
    """
    Get postcode to area look-up table.
    """
    df = get_df_from_url(
        config["data_source"]["postcode_to_area_lookup_2011_url"],
        file_type="csv",
        encoding="latin-1",
        compression="zip",
        header=0,
        sep=",",
        quotechar='"',
    )

    return df


def get_df_rural_class_s():
    """
    Get Scottish postcode directory df.
    """
    rural_df_list = []

    for file_name in ["SmallUser.csv", "LargeUser.csv"]:
        df = get_df_from_url(
            config["data_source"]["scottish_postcode_directory_2020_url"],
            file_type="zip",
            file_name=file_name,
            encoding="latin-1",
        )
        df["scot_rural_urban_classification_2fold"] = df[
            "UrbanRural8Fold2020Code"
        ].apply(lambda x: 1 if x <= 5 else 2)
        df["postcode"] = df["Postcode"].str.replace(" ", "")
        df = df[["postcode", "scot_rural_urban_classification_2fold"]]
        rural_df_list.append(df)
        time.sleep(2)

    rural_class_scotland_df = pd.concat([rural_df_list[0], rural_df_list[1]])

    return rural_class_scotland_df


def generate_df_postcode_to_rural_class(
    postcode_to_area_df,
    rural_df_dict,
    rural_sheet_names=["OA11", "LSOA11", "MSOA11"],
    pcd_join_keys=["OA11CD", "LSOA11CD", "MSOA11CD"],
    rural_class_variable="rural_urban_classification_2011_(2_fold)",
    postcode_col="PCD7",
):
    """
    Join rurality and postcode to area dataframes.
    """
    df = postcode_to_area_df.copy()
    df = _join_df_postcode_to_rural_class(
        df, rural_df_dict, rural_sheet_names, pcd_join_keys
    )
    df[rural_class_variable] = _fill_cols(
        df=df,
        series=df[f"{rural_sheet_names[0]}_{rural_class_variable}"],
        variable=rural_class_variable,
        rural_sheet_names=rural_sheet_names,
    )
    df["postcode"] = df[postcode_col].str.replace(" ", "")

    return df


def _join_df_postcode_to_rural_class(
    df, rural_df_dict, rural_sheet_names, pcd_join_keys
):
    """
    Join each rural classification sheet to postcode to area dataframe
    """
    for sheet, pcd_join in zip(rural_sheet_names, pcd_join_keys):
        rural_df = rural_df_dict[sheet]
        rural_join = rural_df.columns[0]
        df = df.merge(rural_df, how="left", left_on=pcd_join, right_on=rural_join)

    return df


def _fill_cols(
    df, series, variable, rural_sheet_names
):  # TODO: only works effectively when rural_sheet_names in order of OA size
    """
    Fill na values in specified rural urban classification column
    """
    cols = [f"{sheet_name}_{variable}" for sheet_name in rural_sheet_names]
    for col in cols:
        series = series.fillna(df[col])

    return series
