import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import SpectralClustering, KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('./Train.csv')
data.columns = ['id', 'warehouse', 'shipment', 'customer_calls', 'customer_rating', 'cost', 'prior_purchases', 'importance', 'gender','discount','weight', 'ontime']
data.head()

label_encoder = LabelEncoder()
hot_encoder = OneHotEncoder()

data['importance'] = label_encoder.fit_transform(data['importance']) # 1 = low, 2 = medium, 3 = high
data['gender'] = label_encoder.fit_transform(data['gender']) # 0 = female, 1 = male
data['shipment'] = data['shipment'].astype('category')
data['warehouse'] = data['warehouse'].astype('category')
data = pd.get_dummies(data)


X = data[['customer_rating', 'weight', 'shipment_Flight', 'shipment_Road', 'shipment_Ship', 'cost']]
y = data['ontime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# Random Forest Parameters
# See the excel for previous results
max_depth = [2, 4, 10, 20, 40]
random_state = 0
n_estimators = [5, 10, 50, 100, 200, 500]    # number of trees
min_samples_split = 10 # minimum samples to split a node
min_samples_leaf = 3  # minimum samples to be a leaf


for depth in max_depth:
    for estimator in n_estimators:
        random_forest = RandomForestClassifier(max_depth=depth, random_state=random_state, n_estimators=estimator, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(X_train, y_train)

        y_predicted = random_forest.predict(X_test)

        print("Tree number:", estimator, " Depth:", depth)
        print(classification_report(y_test, y_predicted, target_names=["On Time", "Delayed"]))

# Random Forest Parameters
# See the excel for previous results
max_depth = 2
random_state = 0
n_estimators = 50   # number of trees
min_samples_split = [2, 4, 10, 50, 100, 200] # minimum samples to split a node
min_samples_leaf = [1, 5, 8, 10, 20]  # minimum samples to be a leaf


for split in min_samples_split:
    for leaf in min_samples_leaf:
        random_forest = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators, min_samples_split=split, min_samples_leaf=leaf).fit(X_train, y_train)

        y_predicted = random_forest.predict(X_test)

        print("Split:", split, "Leaf:", leaf)
        print(classification_report(y_test, y_predicted, target_names=["On time", "Delayed"]))

