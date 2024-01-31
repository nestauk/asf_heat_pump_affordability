import requests
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from typing import Optional, Union
from asf_heat_pump_affordability import config


def get_df_from_url(
    url: str, extract_file: Optional[str] = None, **kwargs
) -> pd.DataFrame:
    """
    Get dataframe from file stored at URL.

    Args
        url (str): URL location of file download
        extract_file (str): name of file to extract
        **kwargs

    Returns
        pd.DataFrame: dataframe from file
    """
    with requests.Session() as session:
        res = session.get(url)
    if BytesIO(res.content).getvalue()[:4] == bytes(
        "PK\x03\x04", "utf-8"
    ):  # check for zip file signature
        try:
            df = pd.read_excel(BytesIO(res.content), **kwargs)
        except pd.errors.OptionError:
            df = pd.read_csv(
                ZipFile(BytesIO(res.content)).open(name=extract_file), **kwargs
            )
    else:
        df = pd.read_csv(BytesIO(res.content), **kwargs)

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


def get_df_onspd_gb(pcd_col: str = "pcd", ruc_col: str = "ru11ind") -> pd.DataFrame:
    """
    Get ONS postcode directory (ONSPD) for Great Britain.

    Args
        pcd_col (str): name of column containing postcodes
        ruc_col (str): name of column containing rural-urban classification codes

    Returns
        pd.DataFrame: postcode directory for Great Britain
    """
    df = get_df_from_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
    )

    df["postcode"] = df[pcd_col].str.replace(" ", "")
    df["ruc_2fold"] = df[ruc_col].apply(lambda x: _ruc_code_conversion(x))

    return df


def _ruc_code_conversion(ruc_code: Union[str, int, float]):
    """
    Convert rural-urban classification code to 2-fold rural-urban classification; either "rural" or "urban".

    Args
        ruc_code (Union[str, int, float]): rural-urban classification code

    Returns
        str: 2-fold rural-urban classification; "rural" or "urban"
    """
    if isinstance(ruc_code, float) and not np.isnan(
        ruc_code
    ):  # convert RUC codes that appear as float values
        ruc_code = str(int(ruc_code))
    try:
        ruc = config["rural_urban_classification_mapping"][ruc_code]
    except KeyError:
        ruc = np.nan

    return ruc
