# fii_dii.py

import requests
import pandas as pd

def get_fii_dii_data():
    url = "https://www.nseindia.com/api/fiidiiTradeReact"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers)
        data = response.json()

        df = pd.DataFrame(data["data"])
        return df
    except:
        return pd.DataFrame()
