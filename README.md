# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program.

2.Import the required libraries such as pandas, sklearn, and chardet.

3.Detect the encoding format of the spam.csv file.

4.Load the dataset using pandas.

5.Separate the dataset into input (message text) and output (spam or ham label).

6.Split the data into training data and testing data using train_test_split.

7.Convert the text messages into numerical form using CountVectorizer.

8.Train the SVM (Support Vector Machine) model using the training data.

9.Use the trained model to predict whether messages are spam or not.

10.Calculate and display the accuracy of the model.
 

## Program:
```

Program to implement the SVM For Spam Mail Detection..
Developed by: Priya Varshini P
RegisterNumber:  212224240119
import chardet

file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv('spam.csv', encoding='Windows-1252')

data.info()
print(data.isnull().sum())

x = data["v1"].values
y = data["v2"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

print("Predicted values:")
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)



```

## Output:

<img width="451" height="351" alt="image" src="https://github.com/user-attachments/assets/e34bd2f3-0b87-4d32-8d7d-6b32d767183e" />


<img width="340" height="149" alt="image" src="https://github.com/user-attachments/assets/eb134e6b-f6ea-4597-a110-69f331ffe3fa" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
