import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

dataframe = pd.read_csv("diabetes.csv")

feature_cols = ["Pregnancies", "Insulin", "BMI", "Age"]

X = dataframe[feature_cols]
y = dataframe["Outcome"]

# train_test_split splits the entire selected
# dataframe into training and testing data.
# test_size is the percentage we'd like to use
# for testing and the remaining % is the data we'd like
# to use for training
# random-state seeds the selection of which rows to use for
# training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# print how many values we have as either a 1 or 0
print(y_test.value_counts())

# Show that we have 25% of the rows from each dataframe
print(len(y_test) / len(y))
print(len(X_test) / len(X))

# Validate where 25 comes from
print(len(X) * 0.25)
print(len(y) * 0.25)

# Build the logistic regression model
logreg = LogisticRegression()

# Train the logistic regression
logreg.fit(X_train, y_train)

# Using the X_test data, predict the corresponding y values
y_pred = logreg.predict(X_test)
print(y_pred)

# print out the legit values
print(y_test.values.T)

result_dict = {"no_diab_c": 0, "has_diab_c": 0, "has_diab_i": 0, "no_diab_i": 0}


def compare_y_data(y_pred, y_test):
    for i in range(len(y_pred)):
        if y_pred[i] > y_test[i]:
            result_dict["has_diab_i"] += 1
        elif y_pred[i] < y_test[i]:
            result_dict["no_diab_i"] += 1
        elif y_pred[i] == y_test[i] and y_pred[i] == 1:
            result_dict["has_diab_c"] += 1
        else:
            result_dict["no_diab_c"] += 1

    return result_dict


# Compare the y data
results = compare_y_data(y_pred, y_test.values)

# Iterate over the keys and compare
print()
for key in results.keys():
    print(f"{key}: {results[key]}")

confusion = metrics.confusion_matrix(y_test, y_pred)
# Print out the confusion matrix (essentially what was done with the dictionary before)
print(confusion)


def compute_matrix_score(confusion):
    TN = confusion[0][0]
    FP = confusion[0][1]
    FN = confusion[1][0]
    TP = confusion[1][1]

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # The f1 score includes both precision and recall
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


# Compute the matrix score
print(compute_matrix_score(confusion))

# Gives us an accuracy score of our classifier
print(metrics.accuracy_score(y_test, y_pred))

print(confusion[1, 1] / (confusion[0, 1] + confusion[1, 1]))

# Predict the probability that the features of X test result in either true
# or false
y_pred_prob = logreg.predict_proba(X_test)

plt.hist(y_pred_prob[:, 1], bins=8)
plt.show()

print(y_train.value_counts())
# Obtain threshold l
threshold = y_train.value_counts()[1] / len(y_train)
threshold = 0.348000000


def compute_y_labels(y_pred_prob):
    threshold_count = []
    for num in y_pred_prob[:, 1]:
        if num > threshold:
            threshold_count.append(1)
        else:
            threshold_count.append(0)
    return threshold_count


print(compute_y_labels(y_pred_prob))

# Create the new confusion matrix.
new_confusion = metrics.confusion_matrix(y_test, compute_y_labels(y_pred_prob))
print(y_pred_prob[:, 1])

# Print out the old and new confusion matricies.
print(confusion)
print(new_confusion)

# Compute the scores of each matrix.
print(compute_matrix_score(confusion))
print(compute_matrix_score(new_confusion))
