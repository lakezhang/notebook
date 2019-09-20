#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def decode_line(line) :
    items = tf.decode_csv(line, CSV_TYPES, ',')

    features = dict(zip(COLUMN_NAMES, items))
    label = features.pop('Species')

    return features, label

def get_input_fn(file_name, batch_size = 32, is_shuffle = False, epochs=0) :
    dataset = tf.data.TextLineDataset(file_name).skip(1)
    dataset = dataset.map(decode_line)

    if epochs > 0:
        dataset = dataset.repeat(epochs)

    if is_shuffle :
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    return dataset

def main() :

    feature_columns = [tf.feature_column.numeric_column(key=col) for col in COLUMN_NAMES[:-1]]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3)

    steps = int(120 * 3 / 32)

    classifier.train(input_fn = lambda : get_input_fn("/Users/zhangzeming/codes/github/dataset/iris/iris_training.csv"
        , epochs = 3), steps = steps)

    eval_result = classifier.evaluate(
        input_fn = lambda : get_input_fn("/Users/zhangzeming/codes/github/dataset/iris/iris_test.csv"))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

if __name__ == '__main__':
    main()

