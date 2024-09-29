import json
import os
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree as sktree


class Training():

    def __init__(self, data, filename):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), filename), "r") as json_file:
            configuration = json.load(json_file)

        self.data = data
        self.col_x = configuration["col_x"]
        self.col_y = configuration["col_y"]
        self.X = data[self.col_x]
        self.Y = data[self.col_y]
        (self.X_tr,
         self.X_test,
         self.Y_tr,
         self.Y_test) = train_test_split(self.X,
                                         self.Y,
                                         train_size=configuration["train_size"],
                                         random_state=1234,
                                         stratify=data[configuration["stratify_ax"]][configuration["stratify_col"]])
        self.Model = ''

    def show_boxplot(self, x, y):
        sns.boxplot(data=self.data, x=x, y=y)
        plt.show()

    def show_scatter(self, x, y, hue, markers):
        if markers=="":
            sns.scatterplot(data=self.data, x=x, y=y, hue=hue, style=hue, s=150)
        else:
            sns.scatterplot(data=self.data, x=x, y=y, hue=hue, style=hue, markers=markers, s=150)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.show()

    def show_tree(self, figsize, filled):
        plt.figure(figsize=figsize)
        feature_names = list(self.X_tr.columns)
        sktree.plot_tree(self.Model,
                         feature_names=feature_names,
                         filled=filled)
        plt.show()

    def show_features(self):
        importance = self.Model.feature_importances_
        feature_importance = pd.Series(importance, index=self.X_tr.columns)
        feature_importance.sort_values().plot(kind='bar')
        plt.ylabel('Importance')
        plt.show()

    def pre_pruning_classified(self, grid):
        classifier = sktree.DecisionTreeClassifier(random_state=1234)
        gcv = GridSearchCV(estimator=classifier, param_grid=grid)
        gcv.fit(self.X_tr, self.Y_tr)
        pruned_model = gcv.best_estimator_
        print(f"Best params: {pruned_model.fit(self.X_tr, self.Y_tr)}")
        print(f"New model training score: {pruned_model.score(self.X_tr, self.Y_tr)}")
        print(f"New model test score: {pruned_model.score(self.X_test, self.Y_test)}")

    def pre_pruning_regression(self, regressor):
        path = regressor.cost_complexity_pruning_path(self.X_tr, self.Y_tr)
        ccp_alphas = path.ccp_alphas
        ccp_alphas = ccp_alphas[:-1] #remove the 1-node tree from option
        train_scores, test_scores = [], []
        for alpha in ccp_alphas:
            regressor_ = sktree.DecisionTreeRegressor(random_state=1234, ccp_alpha=alpha)
            model_ = regressor_.fit(self.X_tr, self.Y_tr)
            train_scores.append(model_.score(self.X_tr, self.Y_tr))
            test_scores.append(model_.score(self.X_test,self.Y_test))
        return ccp_alphas,train_scores,test_scores

    def dummy_values(self):
        self.X_test = pd.get_dummies(self.X_test)
        self.X_tr = pd.get_dummies(self.X_tr)