
class Preprocessing:

    def encoding(self, dataset, features):
        for i in range(len(features)):
            dataset.iloc[:, i] = [features[i][item] for item in dataset.iloc[:, i].values]
        return dataset
