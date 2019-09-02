import tensorflow as tf 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

iris = pd.read_csv('iris.csv')

print(iris.head())

iris.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

print(iris.head())

iris['target'] = iris['target'].apply(int)

print(iris.head())

X = iris.drop('target', axis = 1)
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

feature_col = []

for col in X.columns:
	feature_col.append(tf.feature_column.numeric_column(col))

input_fn = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 10, num_epochs = 5, shuffle = True)

classifier = tf.estimator.DNNClassifier(hidden_units = [10, 20, 10], n_classes = 3, feature_columns = feature_col)

classifier.train(input_fn = input_fn, steps = 50)

predict_fn = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle = False)

predictions = list(classifier.predict(input_fn = predict_fn))

f_pred = []

for pred in predictions:
	f_pred.append(pred['class_ids'][0])

print(confusion_matrix(y_test, f_pred))
print(classification_report(y_test, f_pred))
