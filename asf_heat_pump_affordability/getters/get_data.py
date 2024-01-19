import requests
import pandas as pd
from io import BytesIO


def get_df_from_url(url):
    """
    Get dataframe from .csv file stored at URL.

    Args
        url (str): URL location of .csv file download

    Returns
        pd.DataFrame
    """
    with requests.Session() as session:
        res = session.get(url)
    df = pd.read_csv(BytesIO(res.content))

    return df
