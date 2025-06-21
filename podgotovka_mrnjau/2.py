import os


os.environ['OPENBLAS_NUM_THREADS'] = '1'
from sklearn.tree import DecisionTreeClassifier

from dataset_script import dataset

def split_dataset(set):
    return [row[:-1] for row in set], [row for row in set]

if __name__ == '__main__':
    P = int(input()) / 100
    C = input()
    L = int(input())

    train_set = dataset[:int(P * len(dataset))]
    test_set = dataset[int(P * len(dataset)):]
    train_X, train_Y = split_dataset(train_set)
    test_X, test_Y = split_dataset(test_set)
    classifierTree = DecisionTreeClassifier(criterion=C, max_leaf_nodes=L, random_state=0)
    classifierTree.fit(train_X, train_Y)
    acc1 = classifierTree.score(test_X, test_Y)
    print("Accuracy (Single Tree):", acc1)

    model1 = DecisionTreeClassifier(criterion=C,max_leaf_nodes=L,random_state=0)
    model2 = DecisionTreeClassifier(criterion=C,max_leaf_nodes=L,random_state=0)
    model3 = DecisionTreeClassifier(criterion=C,max_leaf_nodes=L, random_state=0)

    model1.fit(train_X, [1 if row[-1] == 'Perch' else 0 for row in train_Y])
    model2.fit(train_X, [1 if row[-1] == 'Roach' else 0 for row in train_Y])
    model3.fit(train_X, [1 if row[-1] == 'Bream' else 0 for row in train_Y])



    models = {
        model1:'Perch',
        model2:'Roach',
        model3:'Bream'
    }


    acc2 = 0

    for row in test_set:
        true_class = row[-1]
        print(true_class)
        for m,n in models.items():

            ...




