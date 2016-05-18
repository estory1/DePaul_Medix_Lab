from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
print "----- IRIS DATA -----"
print iris.data
print "----- IRIS TARGET -----"
print iris.target
print "----- IRIS CV SCORE -----"
print cross_val_score(clf, iris.data, iris.target, cv=10)