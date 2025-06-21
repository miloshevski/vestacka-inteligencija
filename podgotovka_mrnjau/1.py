import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from dataset_script import dataset
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

def split_dataset(data):
    return [row[:-1] for row in data], [row[-1] for row in data]

if __name__ == '__main__':
    C = int(input())     # 0 or 1
    P = int(input()) / 100  # percentage to split

    # Feature engineering: combine first and second-to-last column
    new_dataset = [[row[0] + row[-2]] + row[1:10] + [row[-1]] for row in dataset]

    class_good = [row for row in new_dataset if row[-1] == 'good']
    class_bad = [row for row in new_dataset if row[-1] == 'bad']

    if C == 0:
        train_set = class_good[:int(P * len(class_good))] + class_bad[:int(P * len(class_bad))]
        test_set = class_good[int(P * len(class_good)):] + class_bad[int(P * len(class_bad)):]
    else:
        idx = 1 - P
        train_set = class_good[int(idx * len(class_good)):] + class_bad[int(idx * len(class_bad)):]
        test_set = class_good[:int(idx * len(class_good))] + class_bad[:int(idx * len(class_bad))]

    train_X, train_Y = split_dataset(train_set)
    test_X, test_Y = split_dataset(test_set)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train_X)

    # Without normalization
    classifier = GaussianNB()
    classifier.fit(train_X, train_Y)
    acc1 = classifier.score(test_X, test_Y)
    print(acc1)

    # With normalization
    classifier2 = GaussianNB()
    classifier2.fit(scaler.transform(train_X), train_Y)
    acc2 = classifier2.score(scaler.transform(test_X), test_Y)
    print(acc2)
