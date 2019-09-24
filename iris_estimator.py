#!/usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import pandas as pd

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def decode_line(line, need_label = True) :
    '''
        NO need_label when predict
    '''
    items = tf.decode_csv(line, CSV_TYPES, ',')

    features = dict(zip(COLUMN_NAMES, items))
    label = features.pop('Species')

    if not need_label :
        return features

    return features, label

def get_input_fn(file_name, batch_size = 32, is_shuffle = False, epochs=0, label = True) :
    dataset = tf.data.TextLineDataset(file_name).skip(1)
    dataset = dataset.map(lambda x : decode_line(x, label))

    if epochs > 0:
        dataset = dataset.repeat(epochs)

    if is_shuffle :
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    return dataset

def main(_) :

    feature_columns = [tf.feature_column.numeric_column(key=col) for col in COLUMN_NAMES[:-1]]

    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 10],
        n_classes=3,
        optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
        model_dir=FLAGS.model_dir
    )

    # train
    TRAIN_EXAMPLES = 120
    steps = int(TRAIN_EXAMPLES * FLAGS.epochs / FLAGS.batch_size)

    classifier.train(input_fn = lambda : get_input_fn(FLAGS.train_file
        , epochs = FLAGS.epochs, batch_size = FLAGS.batch_size, is_shuffle = True), steps = steps)

    # evaluate
    eval_result = classifier.evaluate(input_fn = lambda : get_input_fn(FLAGS.test_file))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    #predict 
    predictions = classifier.predict(
        input_fn = lambda : get_input_fn(FLAGS.predict_file, batch_size = 10, label=False))

    template = ('Prediction is "{}" ({:.1f}%)')

    count = 0
    for pred_dict in predictions :
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id], 100 * probability))
        count += 1
        if count > 4 :
            break

    #export
    feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

    export_dir = classifier.export_savedmodel(FLAGS.model_dir+'/export', serving_input_receiver_fn)

def server_test(_) :
    #check exported model using saved_model_cli
    #saved_model_cli show --dir export/1569225349/ --tag_set serve --signature_def serving_default

    #saved_model_cli run --dir export/1569225349/ --tag_set serve --signature_def serving_default \
    #                --input_examples 'inputs=[{"SepalLength":[5.1],"SepalWidth":[3.3],"PetalLength":[1.7],"PetalWidth":[0.5]}]'

    export_dir = './model/iris/export/1569225349'
    # 从导出目录中加载模型，并生成预测函数。
    predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

    # 使用 Pandas 数据框定义测试数据。
    inputs = pd.DataFrame({
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    })

    # 将输入数据转换成序列化后的 Example 字符串。
    examples = []
    for index, row in inputs.iterrows():
        feature = {}
        for col, value in row.iteritems():
            feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
        examples.append(example.SerializeToString())

    # 开始预测
    predictions = predict_fn({'inputs': examples})

    print(predictions)

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('model_dir', './model/iris', 'Directory where all models are saved')
    flags.DEFINE_string('train_file', "../dataset/iris/iris_training.csv", 'train file')
    flags.DEFINE_string('test_file', "../dataset/iris/iris_test.csv", 'test file')
    flags.DEFINE_string('predict_file', "../dataset/iris/iris_test.csv", 'predict file')

    flags.DEFINE_integer('batch_size', 32, 'Batch size.')
    flags.DEFINE_integer('epochs', 100, 'Num of batches to train (epochs).')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning Rate')

    FLAGS = flags.FLAGS
    tf.logging.set_verbosity(tf.logging.INFO)

    # train and evaluate
    tf.app.run()
    # test server
    tf.app.run(main=server_test)


