import json
import os
import pandas as pd

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


class Training():

    def __init__(self, data):
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),"config.json"), "r") as json_file:
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

    def dummy_values(self):
        self.X_test = pd.get_dummies(self.X_test)
        self.X_tr = pd.get_dummies(self.X_tr)