import pandas as pd
from training import Training
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree as sktree
from sklearn.metrics import mean_absolute_error


def read():
    return pd.read_csv('income.csv')

if __name__ == '__main__':
    income = read()
    trained = Training(income, "income.json")
    #trained.show_boxplot('Education', 'Salary')
    #trained.show_boxplot(income, 'Education', 'Age')
    #trained.show_scatter('Age','Salary', 'Education',"")
    #print(f"{trained.X_tr.shape}, {trained.X_test.shape}")
    trained.dummy_values()
    #print(f"{trained.X_test.head()}")
    regressor = sktree.DecisionTreeRegressor(random_state=1234)
    trained.Model = regressor.fit(trained.X_tr, trained.Y_tr)
    #print(f"Prediction score {trained.Model.score(trained.X_test, trained.Y_test)}")
    #y_test_predicted = trained.Model.predict(trained.X_test)
    #print(f"Average prediction errors: {mean_absolute_error(trained.Y_test, y_test_predicted)}")
    #trained.show_tree((15,15),True)
    #trained.show_features()
    #print(f"Model score: {trained.Model.score(trained.X_tr, trained.Y_tr)}")
    ccp_a, tr_score, test_score = trained.pre_pruning_regression(regressor)
    #plt.plot(ccp_a, tr_score, marker='o', drawstyle='steps-post')
    #plt.plot(ccp_a, test_score, marker='^', drawstyle='steps-post')
    #plt.show()
    ix = test_score.index((max(test_score)))
    best_alpha = ccp_a[ix]
    regressor = sktree.DecisionTreeRegressor(random_state=1234, ccp_alpha=best_alpha)
    trained.Model = regressor.fit(trained.X_tr, trained.Y_tr)
    print(f"Model score: {trained.Model.score(trained.X_tr, trained.Y_tr)}")
    print(f"Prediction score {trained.Model.score(trained.X_test, trained.Y_test)}")
    #trained.show_tree((15,15),True)
    y_test_predicted = trained.Model.predict(trained.X_test)
    y_test_real = list(trained.Y_test['Salary'])
    print(y_test_predicted)
    print(y_test_real)
    plt.plot(range(1,13,1), y_test_real, marker='^')
    plt.plot(range(1,13,1), y_test_predicted, marker='o')
    plt.show()


