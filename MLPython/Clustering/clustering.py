import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sympy import false


def show_boxplot(data, x, y):
    sns.boxplot(data=data, x=x, y=y)
    plt.show()


def show_scatterplot(x, y, s, data):
    sns.scatterplot(data=data,x=x, y=y, s=s)
    plt.show()

if __name__ == "__main__":
    customers = pd.read_csv("mallcustomers.csv")
    #print(customers.info())
    #print(customers.head())
    #print(customers.describe(include="all"))
    #show_boxplot(customers, "Income", "Gender")
    #show_boxplot(customers, "Age", "Gender")
    #show_boxplot(customers, "SpendingScore", "Gender")
    #show_scatterplot(customers["Age"], customers["Income"], 150)
    #show_scatterplot(customers["Age"], customers["SpendingScore"], 150)
    #show_scatterplot(customers["SpendingScore"], customers["Income"], 150)
    #print(customers[["SpendingScore","Income"]].describe().round(2))

    #Normalization with scaler
    scaler = StandardScaler()
    customers_normalize = scaler.fit_transform(customers[["Income","SpendingScore"]])

    #convert to data frame
    customers_scaled = pd.DataFrame(customers_normalize, columns=["Income","SpendingScore"])
    #print(customers_scaled[["SpendingScore", "Income"]].describe().round(2))

    #creation of clusters by KMeans
    km = KMeans(n_clusters=5, n_init=25, random_state=1234)
    km.fit(customers_scaled)
    #print(km.labels_)

    #wcss for clusters
    print(f"WCSS: {km.inertia_}")

    #create a numpy array to see the cluster sizes
    print(f"Cluster sizes:\n {pd.Series(km.labels_).value_counts().sort_index()}")
    print(f"Cluster centroids:\n {km.cluster_centers_}")
    cluster_centers = pd.DataFrame(km.cluster_centers_,columns=["Income","SpendingScore"])
    print(f"Cluster centers as DF:\n{cluster_centers}")
    sns.scatterplot(data=customers_scaled,
                    x="SpendingScore",
                    y="Income",
                    s=150,
                    hue=km.labels_,
                    alpha=0.8,
                    palette="colorblind",
                    legend=false)
    sns.scatterplot(data=cluster_centers,
                    x="SpendingScore",
                    y="Income",
                    s=600,
                    marker="D",
                    hue=cluster_centers.index,
                    palette="colorblind",
                    legend=false,
                    ec='black')
    for i in range(len(cluster_centers)):
        plt.text(x=cluster_centers.SpendingScore[i],
                 y=cluster_centers.Income[i],
                 s = i,
                 horizontalalignment = 'center',
                 verticalalignment='center',
                 size=15,
                 weight='bold',
                 color='white')
    #plt.show()

    #Finding the optimal number of clusters

    #1. Bend method
    wcss = []
    for k in range(2,11):
        km = KMeans(n_clusters=k, n_init=25, random_state=1234)
        km.fit(customers_scaled)
        wcss.append(km.inertia_)

    wcss_series = pd.Series(wcss, index=range(2,11))
    plt.clf()
    sns.lineplot(y=wcss_series, x=wcss_series.index)
    sns.scatterplot(y=wcss_series, x=wcss_series.index, s=150)
    #plt.show()

    #2.
