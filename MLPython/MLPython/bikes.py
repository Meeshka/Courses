
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def showPlt(df, kind, xlabel, ylabel, title):
    df.plot(kind=kind, x=xlabel, y=ylabel)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == "__main__":
    bikes = pd.read_csv('bikes.csv')
    #print(type(bikes))
    #print(bikes.head())
    #print(bikes.info())
    #print(bikes.describe())

    #showPlt(bikes, "scatter", "temperature", "rentals", "Temperature/Rentals")
    #showPlt(bikes, "scatter", "humidity", "rentals", "Humidity")
    #showPlt(bikes, "scatter", "windspeed", "rentals", "Wind")
    response = "rentals"
    y = bikes[[response]]
    #print(y)
    predictors = list(bikes.columns)
    predictors.remove(response)
    x = bikes[predictors]
    #print(x)
    x_tr, x_test, y_tr, y_test = train_test_split(x,y,random_state=1234)
    #print(f"{x_test.info()}\n{y_test.info()}")
    model = LinearRegression().fit(x_tr, y_tr)
    print(f"{model.score(x_test,y_test)}")

    y_pred = model.predict(x_test)
    print(f"{mean_absolute_error(y_test,y_pred)}")

