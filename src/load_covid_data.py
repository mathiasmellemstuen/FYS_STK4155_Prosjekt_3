import pandas as pd

def load_covid_data():
    """
        Loading data from data/covid_data.csv and parsing content. Changing dates to unix eopch time to
        represent dates in numerical values.

        Returns
        -------
        Array
            Array with column name strings
        np.ndarray
            Two dimensional numpy array with all parsed data
    """
    data = pd.read_csv("data/covid_data.csv", index_col=False, low_memory = False)

    headers = data.columns.tolist()

    data["DATE_DIED"] = pd.to_datetime(data["DATE_DIED"], dayfirst=True)
    data["DATE_DIED"] = pd.to_numeric(data["DATE_DIED"])

    return headers, data.to_numpy()