
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
class ModelSelection:

    def cross_validation(self, clf, X_train, y_train, log_file_name):
        bestMean = -1
        bestScores = []
        bestIndex = -1

        print('5-fold crass validation', file=log_file_name)
        plt.figure()
        lw = 2
        plt.figure(figsize=(15, 15))
        for i in range(1, 20):
            clf.max_depth = i
            k_fold = KFold(n_splits=5)
            scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=k_fold, n_jobs=-1)
            print(i, "Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print(i, "Accuracy: %0.3f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), file=log_file_name)
            if bestMean < scores.mean():
                bestMean = scores.mean()
                bestScores = scores
                bestIndex = i

            plt.plot(range(5), scores, lw=lw,
                     label='accuracy: %0.3f (+/- %0.2f) depth = %d' % (
                         scores.mean(), scores.std() * 2, i))

        plt.plot(range(5), bestScores, lw=lw,
                 color='black', linestyle=':', linewidth=4,
                 label='The Best: accuracy: %0.3f (+/- %0.2f) depth = %d' % (
                     bestScores.mean(), bestScores.std() * 2, bestIndex))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Group')
        plt.ylabel('Score')
        plt.title('K-fold Cross Validation')
        plt.legend(loc="lower right")
        plt.savefig('validation/' + 'validation.png')
        print('Result', file=log_file_name)
        print('Best depth : ', bestIndex, " With Mean Score: ", bestMean,
              file=log_file_name)

        return bestIndex
