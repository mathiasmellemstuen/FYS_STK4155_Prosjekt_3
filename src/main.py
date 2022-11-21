from load_covid_data import load_covid_data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from accuracy_score import accuracy_score


def PredictionNN(X,Y):
    M = 100000 #chosen number of data points
    n = len(Y)
    m = int(n/M)
    random_index = np.random.randint(m)*M   
    new_X = X[random_index:random_index+M]
    new_Y = Y[random_index:random_index+M]
    
    new_X = X[0:M]
    new_Y = Y[0:M]

    train_size = 0.5
    test_size = 1 - train_size
    X_train, X_test, Y_train, Y_test = train_test_split(new_X, new_Y, train_size=train_size,
                                                    test_size=test_size)
    
    model = MLPClassifier(hidden_layer_sizes=(12), activation="relu", solver="adam", max_iter=500)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(Y_test, Y_pred)
    print(accuracy_score(Y_test, Y_pred))

if __name__ == "__main__":

    # Loading covid dataset
    headers, X, Y = load_covid_data()
    PredictionNN(X,Y)
    
