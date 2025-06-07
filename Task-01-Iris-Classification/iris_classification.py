# iris_classification.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# read file from Iris.csv
data_path = "Data/Iris.csv"
df = pd.read_csv(data_path)

# display first rows of data
print('few rows of dataset')
print(df.head())

# display general data information
print('/n dataset information')
print(df.info())

# show the number of samples in each species
print('/n numbers of specimens in each species')
print(df['Species'].value_counts())

# delete the Id column
df_no_id =df.drop(columns=['Id'])

# display dataset without an Id
print('/n dataset after removing Id: ')
print(df_no_id.head())

# separating features and tags
x = df_no_id.drop(columns=['Species']) # features (all columns except Species)
y = df_no_id['Species']

# display the first few rows of x and y
print('/n first few lines of features(X): ')
print(x.head())
print('/n first few lines of features (Y): ')
print(y.head())

# converting species to numeric values
le = LabelEncoder()
y = le.fit_transform(y)

# display classes and some examples of y
print('/n species classes after conversion to numbers: ')
print(list(le.classes_))
print('some examples of y after conversion')
print(y[:5])

# dividing data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# display the size of the data 
print('/n training data size (x_train): ', x_train.shape)
print('training data size (x_test)', x_test.shape)
print('training label size (y_train): ', y_train.shape)
print('size of test labels (y_test): ', y_test.shape)

# standardization of features (correct method after splitting)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train) # on training data 
x_test = scaler.transform(x_test)   # on testing data

# show multiple samples of x_train and x_test after standardization
print('/n some examples of training data after standardization: ')
print(x_train[:5])
print('/n some examples of test data after standardization: ')
print(x_test[:5])

# building and training the KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

# prediction on experimental data
y_pred = knn.predict(x_test)

# display few examples of prediction 
print('/n Some examples of model prediction on experimental data: ')
print(y_pred[:5])

# calculation model accuracy
accuracy = accuracy_score(y_test, y_pred)
print('/n model accuracy: ', accuracy)

# classification report
print('/n classification report')
print(classification_report(y_test, y_pred, target_names=le.classes_))

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# matrix visualization
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png') # save chart
plt.close() # close window
print("/n The confusion matrix diagram was saved as file 'confusion matrix.png'")

# cross-validation
print('/n Cross-Validation results: ')
cv_scores = cross_val_score(knn, x, y, cv=5)
print('Accuracy for each flood: ', cv_scores)
print('Mean accuracy: ', cv_scores.mean() * 100, '%')
print('Standard deviation of accuracy: ', cv_scores.std() * 100, '%')

# prediction for a new sample
print("\nPrediction for new sample:")
new_sample_df = pd.DataFrame([[5.2, 3.4, 1.4, 0.2]], columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])
new_sample_scaled = scaler.transform(new_sample_df)
prediction = knn.predict(new_sample_scaled)
print("New sample:", new_sample_df.values)
print("Predicted species:", le.inverse_transform(prediction)[0])
