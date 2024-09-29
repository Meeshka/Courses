import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree as sktree


class Training():

    def __init__(self, data, col_x, col_y):
        self.col_x = col_x
        self.col_y = col_y
        self.X = data[self.col_x]
        self.Y = data[self.col_y]
        (self.X_tr,
         self.X_test,
         self.Y_tr,
         self.Y_test) = train_test_split(self.X,
                                         self.Y,
                                         train_size=0.8,
                                         random_state=1234,
                                         stratify=self.Y)
        self.Model = ''


def first():
    loan = pd.read_csv('loan.csv')
    return loan


def second(data, x, y):
    ax = sns.boxplot(data=data, x=x, y=y)
    plt.show()


def third(data, x, y):
    ax = sns.scatterplot(data=data, x=x, y=y, hue='Default', style='Default', markers=['^', 'o'], s=150)
    ax = plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.show()


def fourth(loan):
    tr_loan = Training(loan,
                       ['Income', 'Loan Amount'],
                       ['Default'])
    print(f"Number of training and testing elements: {tr_loan.X_tr.shape},{tr_loan.X_test.shape}")
    classifier = sktree.DecisionTreeClassifier(random_state=1234)
    tr_loan.Model = classifier.fit(tr_loan.X_tr, tr_loan.Y_tr)
    print(f"Testing prediction score: {tr_loan.Model.score(tr_loan.X_test, tr_loan.Y_test)}")
    return tr_loan


def fifth(model):
    plt.figure(figsize=(15, 15))
    sktree.plot_tree(model.Model,
                     feature_names=model.col_x,
                     class_names=['No', 'Yes'],
                     filled=True)
    plt.show()


def sixth(model):
    importance = model.Model.feature_importances_
    feature_importance = pd.Series(importance, index=['Income', 'Loan Amount'])
    feature_importance.plot(kind='bar')
    plt.ylabel('Importance')
    plt.show()


def seventh(model):
    print(f"Training data score: {model.Model.score(model.X_tr, model.Y_tr)}")
    print(f"Test data score: {model.Model.score(model.X_test, model.Y_test)}")


def eigth(model):
    grid = {
        'max_depth': [2, 3, 4, 5],
        'min_samples_split': [2, 3, 4, 5],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    classifier = sktree.DecisionTreeClassifier(random_state=1234)
    gcv = GridSearchCV(estimator=classifier, param_grid=grid)
    gcv.fit(model.X_tr, model.Y_tr)
    pruned_model = gcv.best_estimator_
    print(f"Best params: {pruned_model.fit(model.X_tr, model.Y_tr)}")
    print(f"New model training score: {pruned_model.score(model.X_tr, model.Y_tr)}")
    print(f"New model test score: {pruned_model.score(model.X_test, model.Y_test)}")
    return pruned_model


if __name__ == "__main__":
    loan = first()
    #print(f"{loan.head()}\n{loan.info()}\n{loan.describe()}")
    #second(loan, 'Default', 'Income')
    #second(loan, 'Default', 'Loan Amount')
    #third(loan, 'Loan Amount', 'Income')
    train_model = fourth(loan)
    #fifth(train_model)
    #sixth(train_model)
    #seventh(train_model)
    new_model = eigth(train_model)
    plt.figure(figsize=(15, 15))
    sktree.plot_tree(new_model,
                     feature_names=train_model.col_x,
                     class_names=['No', 'Yes'],
                     filled=True)
    plt.show()
