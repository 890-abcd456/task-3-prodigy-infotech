# PRODIGY_DS_03

In this task, I built a Decision Tree classifier using the Diabetes dataset to predict whether a person is diabetic based on medical attributes. This task strengthened my understanding of supervised learning, model training, and performance evaluation.
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols] = df[cols].replace(0, df[cols].median())
X = df.drop('Outcome', axis=1)   # Features
y = df['Outcome']               # Target
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(15,8))
plot_tree(dt, feature_names=X.columns, class_names=['No Diabetes','Diabetes'], filled=True)
plt.show()


print(classification_report(y_test, y_pred))

