import requests
import pandas as pd
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from asf_heat_pump_affordability import config, json_schema


def get_df_from_csv_url(url: str, **kwargs) -> pd.DataFrame:
    """
    Get dataframe from CSV file stored at URL.

    Args
        url (str): URL location of CSV file download
        **kwargs for pandas.read_csv()

    Returns
        pd.DataFrame: dataframe from CSV file
    """
    content = _get_content_from_url(url)
    df = pd.read_csv(content, **kwargs)

    return df


def get_df_from_excel_url(url: str, **kwargs) -> pd.DataFrame:
    """
    Get dataframe from Excel file stored at URL.

    Args
        url (str): URL location of Excel file download
        **kwargs for pandas.read_excel()

    Returns
        pd.DataFrame: dataframe from Excel file
    """
    content = _get_content_from_url(url)
    df = pd.read_excel(content, **kwargs)

    return df


def get_df_from_zip_url(url: str, extract_file: str, **kwargs) -> pd.DataFrame:
    """
    Get dataframe from zip file stored at URL.

    Args
        url (str): URL location of zip file download
        extract_file (str): name of file to extract
        **kwargs for pandas.read_csv()

    Returns
        pd.DataFrame: dataframe from zip file
    """
    content = _get_content_from_url(url)
    df = pd.read_csv(ZipFile(content).open(name=extract_file), **kwargs)

    return df


def _get_content_from_url(url: str) -> BytesIO:
    """
    Get BytesIO stream from URL.
    Args
        url (str): URL
    Returns
        io.BytesIO: content of URL as BytesIO stream
    """
    with requests.Session() as session:
        res = session.get(url)
    content = BytesIO(res.content)

    return content


def get_df_imd_income_deciles_engwal(sheet_name: str = "Income") -> pd.DataFrame:
    """
    Get dataframe of income deprivation rank deciles for LSOAs in England and Wales.

    Args
        sheet_name (str): name of sheet where income deprivation data is stored in file

    Returns
        pd.DataFrame: income deprivation rank deciles by LSOA code for England and Wales
    """
    df = get_df_from_excel_url(
        config["data_source"]["engwal_imd_income_url"], sheet_name=sheet_name
    )
    df["engwal_imd_income_rank_decile"] = pd.qcut(
        df["Income Domain Rank (where 1 is most deprived)"],
        10,
        labels=np.arange(1, 11, 1),
    )
    df = df[
        [
            "LSOA Code (2011)",
            "Income Domain Rank (where 1 is most deprived)",
            "engwal_imd_income_rank_decile",
        ]
    ]
    return df


def get_df_imd_income_deciles_sct(
    sheet_name: str = "SIMD 2020v2 ranks",
) -> pd.DataFrame:
    """
    Get dataframe of income deprivation rank deciles for Data Zones in Scotland.

    Args
        sheet_name (str): name of sheet where income deprivation data is stored in file

    Returns
        pd.DataFrame: income deprivation rank deciles by Data Zone for Scotland
    """
    df = get_df_from_excel_url(
        config["data_source"]["sct_imd_income_url"], sheet_name=sheet_name
    )
    df["sct_imd_income_rank_decile"] = pd.qcut(
        df["SIMD2020v2_Income_Domain_Rank"], 10, labels=np.arange(1, 11, 1)
    )
    df = df[
        ["Data_Zone", "SIMD2020v2_Income_Domain_Rank", "sct_imd_income_rank_decile"]
    ]
    return df


def get_list_off_gas_postcodes(sheet_name: str = "Off-Gas Postcodes 2023") -> list:
    """
    Get list of off-gas postcodes in Great Britain.
    Args
        sheet_name (str): name of sheet where off-gas postcode data is stored in file
    Returns
        list: off-gas postcodes in Great Britain
    """
    df = get_df_from_excel_url(
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
    df = get_df_from_zip_url(
        url=config["data_source"]["gb_ons_postcode_dir_url"],
        extract_file=config["data_source"]["gb_ons_postcode_dir_file_path"],
        dtype=json_schema["onspd_data"],
    )

    df["postcode"] = df[pcd_col].str.replace(" ", "")
    df["ruc_2fold"] = df[ruc_col].apply(lambda x: _ruc_code_conversion(x))

    return df


def _ruc_code_conversion(ruc_code: str) -> str:
    """
    Convert rural-urban classification code to 2-fold rural-urban classification; either "rural" or "urban".

    Args
        ruc_code (Union[str, int, float]): rural-urban classification code

    Returns
        str: 2-fold rural-urban classification; "rural" or "urban"
    """
    try:
        ruc = config["rural_urban_classification_mapping"][ruc_code]
    except KeyError:
        ruc = np.nan

    return ruc
