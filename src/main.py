from load_covid_data import load_covid_data
import numpy as np

if __name__ == "__main__":

    # Loading covid dataset
    headers, data = load_covid_data()
    #print(headers, data)
    died = np.where(data[0:100,4] == -2208988800000000000, 0, 1)
    print(data[0:100,4])
    print(died)