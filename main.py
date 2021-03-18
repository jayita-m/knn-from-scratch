import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load iris data and store in dataframe

from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()

# Separate X and y data

X = df.drop('target', axis=1)
y = df.target


# Calculate distance between two points

def minkowski_distance(a, b, p=1):
    # Store the number of dimensions
    dim = len(a)

    # Set initial distance to 0
    distance = 0

    # Calculate minkowski distance using parameter p
    for d in range(dim):
        distance += abs(a[d] - b[d]) ** p

    distance = distance ** (1 / p)

    return distance


# Test the function

minkowski_distance(a=X.iloc[0], b=X.iloc[1], p=1)

# Define an arbitrary test point

test_pt = [4.8, 2.7, 2.5, 0.7]

# Calculate distance between test_pt and all points in X

distances = []

for i in X.index:
    distances.append(minkowski_distance(test_pt, X.iloc[i]))

df_dists = pd.DataFrame(data=distances, index=X.index, columns=['dist'])
df_dists.head()

# Find the 5 nearest neighbors

df_nn = df_dists.sort_values(by=['dist'], axis=0)[:5]
df_nn

# Create counter object to track the labels

counter = Counter(y[df_nn.index])

# Get most common label of all the nearest neighbors

counter.most_common()[0][0]

# Split the data - 75% train, 25% test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                   random_state=1)

# Scale the X data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def knn_predict(X_train, X_test, y_train, y_test, k, p):
    # Counter to help with label voting
    from collections import Counter

    # Make predictions on the test data
    # Need output of 1 prediction per test data point
    y_hat_test = []

    for test_point in X_test:
        distances = []

        for train_point in X_train:
            distance = minkowski_distance(test_point, train_point, p=p)
            distances.append(distance)

        # Store distances in a dataframe
        df_dists = pd.DataFrame(data=distances, columns=['dist'],
                                index=y_train.index)

        # Sort distances, and only consider the k closest points
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

        # Create counter object to track the labels of k closest neighbors
        counter = Counter(y_train[df_nn.index])

        # Get most common label of all the nearest neighbors
        prediction = counter.most_common()[0][0]

        # Append prediction to output list
        y_hat_test.append(prediction)

    return y_hat_test


# Make predictions on test dataset
y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k=5, p=1)

print(y_hat_test)

# Get test accuracy score

print(accuracy_score(y_test, y_hat_test))

# Obtain accuracy score varying k from 1 to 99

accuracies = []

for k in range(1,100):
    y_hat_test = knn_predict(X_train, X_test, y_train, y_test, k, p=1)
    accuracies.append(accuracy_score(y_test, y_hat_test))

# Plot the results

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(range(1,100), accuracies)
ax.set_xlabel('# of Nearest Neighbors (k)')
ax.set_ylabel('Accuracy (%)');
plt.show();