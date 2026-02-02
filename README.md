# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Necessary Libraries and Load Data.
2. Split Dataset into Training and Testing Sets.
3. Train the Model Using Stochastic Gradient Descent (SGD).
4. Make Predictions and Evaluate Accuracy.
5. Generate Confusion Matrix.

## Program:
```python
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: A.Nabithra 
RegisterNumber: 212224230172
*/
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Load the dataset
iris = load_iris()
```

```python
# Create pandas dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())
```
<img width="624" height="242" alt="image" src="https://github.com/user-attachments/assets/e40dde07-8ece-4b27-b01b-edde7c7776b9" />

```python
x = df.drop('target', axis=1)
y = df['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(x_train, y_train)
```
<img width="349" height="90" alt="image" src="https://github.com/user-attachments/assets/fe5adaa4-f89e-4456-85d1-0fdc0969e097" />

```python
cf = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cf)
```
## Output:
<img width="238" height="72" alt="image" src="https://github.com/user-attachments/assets/06168c74-0607-4f14-bae7-76bb9cf8dc2b" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
