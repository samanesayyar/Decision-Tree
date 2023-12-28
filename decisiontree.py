
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from final.ModelSelection import ModelSelection
from final.Preprocessing import Preprocessing
from final.Prune import Prune
from final.Visualize import Visualize

features = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
classes = [1, 2, 3, 4]
log = open('log.txt', 'w')
dataset = pd.read_csv('car.data')

processing = Preprocessing()
visualize = Visualize()
prune = Prune()
modelSelection = ModelSelection()

# buying = ["low", "med", "high", "vhigh"]
# maint = ["low", "med", "high", "vhigh"]
# doors = ["2", "3", "4", "5more"]
# persons = ["2", "4", "more"]
# lug_boot = ["small", "med", "big"]
# safety = ["low", "med", "high"]
# classes = ["vgood", "good", "acc", "unacc"]

measures = [{'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
            {'low': 1, 'med': 2, 'high': 3, 'vhigh': 4},
            {"2": 2, "3": 3, "4": 4, "5more": 5},
            {"2": 2, "4": 4, "more": 5},
            {"small": 1, "med": 2, "big": 3},
            {"low": 1, "med": 2, "high": 3},
            {"vgood": 1, "good": 2, "acc": 3, "unacc": 4}]

dataset = processing.encoding(dataset, measures)
dataset.to_csv('newcar.csv', index=None, header=None)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0)

# SECTIION A
print('Section A', file=log)
classifier = DecisionTreeClassifier(criterion='entropy', max_features=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

visualize.save_tree(tree=classifier, fn="decision_tree_section_A", features=features)
visualize.save_roc(X_train=X_train, y_train=y_train, classifier=classifier, title='ROC Of Decision Tree SECTION A',
                   classes=classes)

print('confusion_matrix: ', file=log)
print(confusion_matrix(y_test, y_pred, labels=classes), file=log)
print('classification report: ', file=log)
print(classification_report(y_test, y_pred), file=log)
# END A

# SECTION B 1
# PRUNING
print('Section B 1', file=log)
classifier = DecisionTreeClassifier(criterion='entropy', max_features=1)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

prune.prune_index(classifier.tree_, 0, 5)
visualize.save_tree(classifier, fn="decision_tree_after_pruning", features=features)

print('confusion_matrix: ', file=log)
print(confusion_matrix(y_test, y_pred, labels=classes), file=log)
print('classification report: ', file=log)
print(classification_report(y_test, y_pred), file=log)

visualize.save_roc(X_train, y_train, classifier, 'ROC Of Decision Tree After Pruning', doPrune=True, classes=classes)

# END B 1

# SECTION B 2
# CROSS VALIDATION
max_depth = modelSelection.cross_validation(DecisionTreeClassifier(criterion='entropy'), X_train, y_train, log)
classifier = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

visualize.save_tree(classifier, fn="decision_tree_after_cross_validation", features=features)
visualize.save_roc(X_train, y_train, classifier, 'ROC Of Decision Tree After Cross Validation', classes=classes)

print('confusion_matrix: ', file=log)
print(confusion_matrix(y_test, y_pred, labels=classes), file=log)
print('classification report: ', file=log)
print(classification_report(y_test, y_pred), file=log)
# END B 2
