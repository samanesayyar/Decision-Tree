
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.tree import export_graphviz
import subprocess
import matplotlib.pyplot as plt

from final.Prune import Prune

class Visualize:

    def save_tree(self, tree, fn="dt", features=[]):
        dotfile = "tree/" + fn + ".dot"
        pngfile = "tree/" + fn + ".png"

        with open(dotfile, 'w') as f:
            export_graphviz(tree, out_file=f,
                            feature_names=features)

        command = ["dot", "-Tpng", dotfile, "-o", pngfile]
        try:
            subprocess.check_call(command)
        except:
            exit("Could not run dot, ie graphviz, "
                 "to produce visualization")

    def save_roc(self, X_train, y_train, classifier, title, doPrune=False, classes=0):
        y_train = label_binarize(y_train, classes=classes)
        n_classes = y_train.shape[1]

        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.10,
                                                                        random_state=0)

        classifier.fit(X_train, y_train)

        if doPrune:
            prune = Prune()
            prune.prune_index(classifier.tree_, 0, 5)

        y_score = classifier.predict(X_validation)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_validation[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute average ROC curve and ROC area
        fpr["average"], tpr["average"], _ = roc_curve(y_validation.ravel(), y_score.ravel())
        roc_auc["average"] = auc(fpr["average"], tpr["average"])

        plt.figure()
        lw = 2
        plt.figure(figsize=(12, 8))
        plt.plot(fpr["average"], tpr["average"],
                 label='average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["average"]),
                 color='green', linestyle=':', linewidth=4)

        for i in range(0, n_classes):
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i + 1, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', color='red', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.annotate('Random Guess', (.5, .48), color='red')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.savefig('roc/' + title + '.png')
