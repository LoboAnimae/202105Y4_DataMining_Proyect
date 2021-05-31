from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, plot_roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pickle import dump
# import pandas_profiling as pdp


data = pd.read_csv('./Train.csv')
label_encoder = LabelEncoder()
hot_encoder = OneHotEncoder()

data['Product_importance'] = label_encoder.fit_transform(
    data['Product_importance'])  # 1 = low, 2 = medium, 3 = high
data['Gender'] = label_encoder.fit_transform(
    data['Gender'])  # 0 = female, 1 = male

tolerance = 0.000001
flight = data[data['Mode_of_Shipment'] == 'Flight']
ship = data[data['Mode_of_Shipment'] == 'Ship']
road = data[data['Mode_of_Shipment'] == 'Road']


def process_logistic_regression(model, name):
    X = model[
        [
            'Customer_care_calls',
            'Cost_of_the_Product',
            'Prior_purchases',
            'Product_importance',
            'Discount_offered',
            'Weight_in_gms',
        ]
    ]
    y = model['Reached.on.Time_Y.N']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    logistic_regression = LogisticRegression(
        random_state=42, tol=tolerance).fit(X_train, y_train)
    y_predicted = logistic_regression.predict(X_test)
    print(f"CHECKING FOR {name}")
    print(classification_report(y_test, y_predicted,
                                target_names=["On Time", "Delayed"]))
    file_name = f"{name}-model.pkl"
    print(f'Saving model {name} as {file_name}')
    plot_roc_curve(logistic_regression, X_test, y_test)
    plt.title(f"Graph for {name}")
    plt.show()

    with open(file_name, 'wb') as f:
        dump(logistic_regression, f)
        print('Saved file')


process_logistic_regression(flight, 'FLIGHT')
process_logistic_regression(ship, 'SHIP')
process_logistic_regression(road, 'ROAD')
