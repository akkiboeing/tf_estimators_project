import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

bank = pd.read_csv('bank_note_data.csv')

print(bank.head())

sns.countplot(x = 'Class', data = bank)
plt.show()

sns.pairplot(bank, hue = 'Class', palette = 'coolwarm', diag_kind = 'hist')
plt.show()

scaler = StandardScaler()

scaler.fit(bank.drop('Class',axis=1))
scaled_feat = scaler.fit_transform(bank.drop('Class', axis = 1))

df = pd.DataFrame(scaled_feat, columns = bank.columns[:-1])
df.head()

X = df
y = bank['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

feature_col = []

for col in X.columns:
    feature_col.append(tf.feature_column.numeric_column(col))

classifier = tf.estimator.DNNClassifier(hidden_units = [10, 10, 10], n_classes = 2, feature_columns = feature_col)

input_fn = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 20, num_epochs = 5,shuffle = True)

classifier.train(input_fn = input_fn, steps = 500)    

pred_fn = tf.estimator.inputs.pandas_input_fn(x = X_test, batch_size = len(X_test), shuffle = False)

predictions = list(classifier.predict(input_fn = pred_fn))

f_pred = []

for pred in predictions:
    f_pred.append(pred['class_ids'][0])

con_mat = confusion_matrix(y_test, f_pred)
print(con_mat)    
print(classification_report(y_test, f_pred))

if((con_mat[0][1] == 0) & (con_mat[1][0] == 0)):
   print('Prefect Accuracy! Overfitting? Maybe.\n')