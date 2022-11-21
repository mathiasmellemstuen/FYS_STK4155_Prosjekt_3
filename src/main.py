from load_covid_data import load_covid_data
import numpy as np

if __name__ == "__main__":

    # Loading covid dataset
    headers, data = load_covid_data()
    print(headers, data)