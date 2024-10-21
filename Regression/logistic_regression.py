import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def show_boxplot(data, x, y):
    sns.boxplot(data=data, x=x, y=y)
    plt.show()


def show_scatterplot(x, y, s):
    sns.scatterplot(x=x, y=y, s=s)
    plt.show()


if __name__ == "__main__":
    loan = pd.read_csv("loan.csv")
    # print(loan.head())
    # print(loan.info())
    # print(loan.describe())

    # show_boxplot(loan, "Default", "Income")
    # show_boxplot(loan, "Default", "Loan Amount")
    # show_scatterplot(loan["Income"], np.where(loan['Default'] == 'No', 0, 1), 150)
    # show_scatterplot(loan["Loan Amount"], np.where(loan['Default'] == 'No', 0, 1), 150)

    y = loan['Default']
    x = loan[['Income', 'Loan Amount']]
    x_tr, x_test, y_tr, y_test = train_test_split(x, y, stratify=y, random_state=123, train_size=0.7)

    classifier = LogisticRegression()
    model = classifier.fit(x_tr, y_tr)
    predict = model.predict(x_test)
    #print(f"{predict}}")
    score = model.score(x_test, y_test)
    #print(score)
    confusion = confusion_matrix(y_test, predict)
    #print(confusion)

    #Beta 0
    #print(model.intercept_)

    #Beta 1...N
    #print(model.coef_)

    log_odds = np.round(model.coef_[0],2)
    #print(log_odds)

    #new_matrix = pd.DataFrame({'log odds': log_odds}, index = x.columns)
    odds = np.round(np.exp(log_odds),2)
    new_matrix = pd.DataFrame({'log odds': odds}, index=x.columns)
    print(new_matrix)
    #            log odds
    #Income           0.36
    #Loan Amount      1.16
    #for each 1$ increase in income, the odds to return the borrowed money decrease on (1-0.36=64%)
    #for each 1$ increase in borrowed summ the odds to return increase 16%

