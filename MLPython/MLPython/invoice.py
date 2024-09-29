import pandas as pd
from training import Training
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree as sktree


def read():
    return pd.read_csv('income.csv')

if __name__ == '__main__':
    income = read()
    trained = Training(income)
    #trained.show_boxplot('Education', 'Salary')
    #trained.show_boxplot(income, 'Education', 'Age')
    #trained.show_scatter('Age','Salary', 'Education',"")
    #print(f"{trained.X_tr.shape}, {trained.X_test.shape}")
    trained.dummy_values()
    #print(f"{trained.X_test.head()}")
    regressor = sktree.DecisionTreeRegressor(random_state=1234)
    trained.Model = regressor.fit(trained.X_tr, trained.Y_tr)
    print(f"{trained.Model.score(trained.X_test, trained.Y_test)}")