import os

from sklearn.metrics import accuracy_score, precision_score
from dataset6 import dataset
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

def split_dataset(x, dataset):
    train_set = dataset[:len(dataset) - x]
    train_X = [row[:-1] for row in train_set]
    train_Y = [row[-1] for row in train_set]

    test_set = dataset[len(dataset) - x:]
    test_X = [row[:-1] for row in test_set]
    test_Y = [row[-1] for row in test_set]

    return train_X, train_Y, test_X, test_Y

def calculate_acc_prec(test_X, test_Y, classifier: DecisionTreeClassifier):
    tp,fp,tn,fn = 0,0,0,0
    predictions = classifier.predict(test_X)

    for pred, actual in zip(predictions, test_Y):
        if actual == "1":
            if pred == actual:
                tp += 1
            else:
                fp += 1
        else:
            if pred == actual:
                tn += 1
            else:
                fn += 1
    prec = 0.0
    if (tp + fp) != 0:
        prec = tp / (tp + fp)

    acc = (tp + tn) / (tp + fp + tn + fn)

    return acc, prec



if __name__=='__main__':
    X = int(input())
    criteria = 'gini'
    classifier = DecisionTreeClassifier(criterion=criteria, random_state=0)
    encoder = OrdinalEncoder()
    encoder.fit([row[:-1] for row in dataset])

    train_X, train_Y, test_X, test_Y = split_dataset(X, dataset)
    train_X = encoder.transform(train_X)
    test_X = encoder.transform(test_X)

    classifier.fit(train_X, train_Y)

    print(f'Accuracy: {accuracy_score(test_Y, classifier.predict(test_X))}')
    print(f"Precision: {precision_score(test_Y,classifier.predict(test_X),pos_label='1',zero_division=0)}")