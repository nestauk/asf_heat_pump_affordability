import requests
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from typing import Optional
import time

from asf_heat_pump_affordability import config, json_schema


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
        df = pd.read_csv(ZipFile(BytesIO(res.content)).open(file_name), **kwargs)
    else:
        df = pd.read_excel(BytesIO(res.content), **kwargs)

    return df


def get_list_off_gas_postcodes(sheet_name: str = "Off-Gas Postcodes 2023") -> list:
    """
    Get list of off-gas postcodes in Great Britain.

    Args
        sheet_name (str): name of sheet where off-gas postcode data is stored in file

    Returns
        list: off-gas postcodes in Great Britain
    """
    df = get_df_from_url(
        url=config["data_source"]["gb_off_gas_postcodes_url"],
        sheet_name=sheet_name,
    )

    off_gas_postcodes_list = df["Post Code"].str.replace(" ", "").to_list()

    return off_gas_postcodes_list


def get_df_rural_class_engwal(sheet_name: str = "OA11") -> pd.DataFrame:
    """
    Get dataframe with rural urban classification information by area for England and Wales.

    Args
        sheet_name (str): name of sheet where rural/urban classification information stored for chosen geographic level.
                          Default "OA11" to get rural/urban classification by output area.

    Returns
        pd.DataFrame: rural urban classification by area for England and Wales
    """
    df = get_df_from_url(
        config["data_source"]["engwal_rural_urban_classification_2011_url"],
        engine="odf",
        sheet_name=sheet_name,
        skiprows=2,
    )

    df.columns = [
        f"{sheet_name}_{'_'.join(col.lower().split(' '))}" for col in df.columns
    ]

    return df


def get_df_postcode_to_area_engwal(postcode_col: str = "PCD7") -> pd.DataFrame:
    """
    Get postcode to area look-up table for England and Wales.

    Args
        postcode_col (str): name of column containing postcodes

    Returns
        pd.DataFrame: postcode to area look-up for England and Wales
    """
    df = get_df_from_url(
        config["data_source"]["engwal_postcode_to_area_lookup_2011_url"],
        file_type="csv",
        encoding="latin-1",
        compression="zip",
        dtype=json_schema["engwal_postcode_to_area_lookup_2011_data"],
    )

    df["postcode"] = df[postcode_col].str.replace(" ", "")

    return df


def get_df_rural_class_sct() -> pd.DataFrame:
    """
    Get rural/urban classification information by postcode for Scotland.

    Returns
        pd.DataFrame: rural/urban classification by postcode for Scotland
    """
    schemas = [
        json_schema["scottish_postcode_directory_2020_small_user_data"],
        json_schema["scottish_postcode_directory_2020_large_user_data"],
    ]
    rural_df_list = []

    for file_name, schema in zip(["SmallUser.csv", "LargeUser.csv"], schemas):
        df = get_df_from_url(
            config["data_source"]["scottish_postcode_directory_2020_url"],
            file_type="zip",
            file_name=file_name,
            encoding="latin-1",
            dtype=schema,
        )
        df = _get_df_rural_class_2fold_sct(df)
        rural_df_list.append(df)
        time.sleep(2)

    df = pd.concat([rural_df_list[0], rural_df_list[1]])

    return df


def _get_df_rural_class_2fold_sct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get 2-fold rural/urban classification for Scotland by postcode.

    Args
        df (pd.DataFrame): dataframe containing rural/urban classification information by postcode for Scotland

    Returns
        pd.DataFrame: dataframe with 2-fold rural/urban classification by postcode for Scotland
    """
    df["sct_rural_urban_class_2fold"] = df["UrbanRural8Fold2020Code"].apply(
        lambda x: "Urban" if x <= 5 else "Rural"
    )
    df["postcode"] = df["Postcode"].str.replace(" ", "")
    df = df.rename(columns={"OutputArea2011Code": "OA11CD"})[
        ["postcode", "OA11CD", "sct_rural_urban_class_2fold"]
    ]

    return df
